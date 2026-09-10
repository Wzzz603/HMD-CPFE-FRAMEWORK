#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_9_train_lasso_bcc_hcp_FIXED.py

Correct classical weighted Lasso Regression baseline for Step-10.

IMPORTANT
---------
This file DOES NOT call 10_7_train.py or 10_6_trainer.py.
Therefore Lasso is NOT trained with Adam/AdamW epochs.

For each phase:
    Step-08 H5
      -> iter_baseline_point_chunks()
      -> train-only normalization
      -> streamed weighted sufficient statistics
      -> coordinate-descent Lasso, one independent solve per stress component
      -> final train/validation evaluation
      -> best_checkpoint.pt / last_checkpoint.pt

Model form:
    y_hat = W x + b

Objective exactly matching 10_5_loss_and_metrics.py:
    weighted_MSE + LASSO_ALPHA * sum(abs(W))

where:
    weighted_MSE =
        sum_i,c w_i * (y_ic - yhat_ic)^2
        ---------------------------------
        sum_i w_i * TARGET_DIM

Bias/intercept is NOT regularized.

Because the six output stress components are separable, this script solves six
independent weighted Lasso problems that share the same weighted Gram matrix.

The exact per-output objective after multiplying the full Step-10 objective by
TARGET_DIM is:
    SSE_c / sum(w) + LASSO_ALPHA * TARGET_DIM * ||W_c||_1

Coordinate descent therefore uses soft-threshold:
    threshold = 0.5 * LASSO_ALPHA * TARGET_DIM

Normalization is model-independent. This script preferentially reuses the
already-computed compatible Linear normalization for the same data/phase/
multiplier.

Put this script in the same folder as:
    10_1_config.py
    10_2_h5_dataset.py
    10_3_normalizer.py
    10_4_models.py
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


# =============================================================================
# 0. USER CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class LauncherConfig:
    PHASES: Tuple[str, ...] = ("BCC", "HCP")
    MULTIPLIER: int = 1

    # None -> use LASSO_ALPHA from 10_1_config.py.
    LASSO_ALPHA_OVERRIDE: float | None = None

    DEVICE: str = "cpu"

    # Prefer the correct Linear normalization for the same H5/split.
    REUSE_LINEAR_NORMALIZATION: bool = True

    # Do not mix with old epoch/AdamW Lasso results.
    ALLOW_OVERWRITE_EXISTING_RESULTS: bool = False

    NORMALIZATION_OVERWRITE: bool = False
    CHECK_ONLY: bool = False

    # Print coordinate-descent progress every N iterations.
    PRINT_EVERY_ITERATIONS: int = 25


RUN = LauncherConfig()


# =============================================================================
# 1. Local imports
# =============================================================================

_THIS_DIR = Path(__file__).resolve().parent


def _import_local(name: str, filename: str):
    path = _THIS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"{filename} must be in the same directory as this script: {path}"
        )

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_cfg_mod = _import_local("step10_cfg_for_lasso_fixed", "10_1_config.py")
_data_mod = _import_local("step10_data_for_lasso_fixed", "10_2_h5_dataset.py")
_norm_mod = _import_local("step10_norm_for_lasso_fixed", "10_3_normalizer.py")
_model_mod = _import_local("step10_model_for_lasso_fixed", "10_4_models.py")

BASE_CFG = _cfg_mod.CFG
INPUT_DIM = int(_cfg_mod.INPUT_DIM)
TARGET_DIM = int(_cfg_mod.TARGET_DIM)
TARGET_ORDER = tuple(_cfg_mod.TARGET_ORDER)

inspect_phase_h5 = _data_mod.inspect_phase_h5
load_lcs_reference = _data_mod.load_lcs_reference
iter_baseline_point_chunks = _data_mod.iter_baseline_point_chunks

compile_train_only_normalization = _norm_mod.compile_train_only_normalization
load_normalization_stats = _norm_mod.load_normalization_stats

build_model = _model_mod.build_model
save_model_structure = _model_mod.save_model_structure


# =============================================================================
# 2. Utilities
# =============================================================================

def _now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Progress:
    def __init__(self, label: str, total: int) -> None:
        self.label = str(label)
        self.total = max(1, int(total))
        self.current = 0
        self.started = time.perf_counter()

    def update(self, amount: int) -> None:
        self.current = min(
            self.total,
            self.current + int(amount),
        )

        ratio = self.current / float(self.total)
        width = 32
        filled = min(width, int(round(width * ratio)))
        bar = "#" * filled + "-" * (width - filled)

        elapsed = time.perf_counter() - self.started
        rate = self.current / max(elapsed, 1.0e-12)
        remaining = (
            (self.total - self.current) / rate
            if rate > 0.0
            else float("inf")
        )

        sys.stdout.write(
            f"\r{self.label:<16s} "
            f"[{bar}] {100.0*ratio:6.2f}% "
            f"{self.current:,}/{self.total:,} points "
            f"| {rate:,.0f} point/s "
            f"| ETA {remaining:,.1f}s"
        )
        sys.stdout.flush()

        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _selected_point_count(
    cfg,
    split: str,
    include_first: bool,
) -> int:
    info = inspect_phase_h5(cfg)

    if split == "train":
        ntraj = int(info.train_count)
        max_traj = cfg.MAX_TRAIN_TRAJECTORIES
        max_points = cfg.BASELINE_MAX_TRAIN_POINTS
    elif split == "val":
        ntraj = int(info.val_count)
        max_traj = cfg.MAX_VAL_TRAJECTORIES
        max_points = cfg.BASELINE_MAX_VAL_POINTS
    else:
        raise ValueError(split)

    if max_traj is not None:
        ntraj = min(ntraj, int(max_traj))

    ntime = int(info.sequence_length)
    if not include_first:
        ntime -= 1

    total = ntraj * ntime

    if max_points is not None:
        total = min(total, int(max_points))

    return int(total)


def _normalize_x(x, stats, cfg):
    x = np.asarray(x, dtype=np.float64)
    if not cfg.NORMALIZE_INPUTS:
        return x
    return (
        x
        - np.asarray(
            stats.input_mean,
            dtype=np.float64,
        )[None, :]
    ) / np.asarray(
        stats.input_std,
        dtype=np.float64,
    )[None, :]


def _normalize_y(y, stats, cfg):
    y = np.asarray(y, dtype=np.float64)
    if not cfg.NORMALIZE_TARGETS:
        return y
    return (
        y
        - np.asarray(
            stats.target_mean,
            dtype=np.float64,
        )[None, :]
    ) / np.asarray(
        stats.target_std,
        dtype=np.float64,
    )[None, :]


def _denormalize_y(y, stats, cfg):
    y = np.asarray(y, dtype=np.float64)
    if not cfg.NORMALIZE_TARGETS:
        return y
    return (
        y
        * np.asarray(
            stats.target_std,
            dtype=np.float64,
        )[None, :]
        + np.asarray(
            stats.target_mean,
            dtype=np.float64,
        )[None, :]
    )


def _weights(raw, cfg):
    raw = np.asarray(
        raw,
        dtype=np.float64,
    ).reshape(-1)

    w = (
        raw
        if cfg.USE_SAMPLE_WEIGHT
        else np.ones_like(raw)
    )

    if not np.isfinite(w).all():
        raise ValueError(
            "Regression weights contain NaN/Inf"
        )
    if np.any(w <= 0.0):
        raise ValueError(
            "Regression weights must be positive"
        )

    return w


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def _check_clean_output(cfg) -> None:
    conflicts = (
        Path(cfg.BEST_CHECKPOINT_PATH),
        Path(cfg.LAST_CHECKPOINT_PATH),
        Path(cfg.HISTORY_CSV_PATH),
        Path(cfg.TRAINING_SUMMARY_PATH),
    )
    existing = [
        path
        for path in conflicts
        if path.exists()
    ]

    if (
        existing
        and not RUN.ALLOW_OVERWRITE_EXISTING_RESULTS
    ):
        raise FileExistsError(
            "Existing Lasso results were found. They may belong to "
            "the old epoch/AdamW implementation.\n"
            "Rename/remove the old Lasso experiment folder first, "
            "or set ALLOW_OVERWRITE_EXISTING_RESULTS=True.\n"
            + "\n".join(
                f"  - {path}"
                for path in existing
            )
        )


# =============================================================================
# 3. Normalization reuse
# =============================================================================

def _prepare_normalization(cfg):
    lasso_path = Path(cfg.NORMALIZATION_PATH)

    if (
        lasso_path.is_file()
        and not RUN.NORMALIZATION_OVERWRITE
    ):
        try:
            stats = load_normalization_stats(
                lasso_path,
                cfg,
            )
        except Exception as exc:
            print(
                "[NORMALIZATION] Existing Lasso file incompatible:",
                exc,
            )
        else:
            print(
                "[NORMALIZATION] Reusing Lasso normalization:",
                lasso_path,
            )
            return stats

    if (
        RUN.REUSE_LINEAR_NORMALIZATION
        and not RUN.NORMALIZATION_OVERWRITE
    ):
        linear_cfg = dataclasses.replace(
            cfg,
            MODEL_TYPE="linear",
            RESUME_MODE="never",
            ENABLE_LIVE_DASHBOARD=False,
            DASHBOARD_SHOW_WINDOW=False,
        )
        linear_path = Path(
            linear_cfg.NORMALIZATION_PATH
        )

        if linear_path.is_file():
            try:
                load_normalization_stats(
                    linear_path,
                    linear_cfg,
                )
            except Exception as exc:
                print(
                    "[NORMALIZATION] Linear file found but incompatible:",
                    exc,
                )
            else:
                lasso_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                shutil.copy2(
                    linear_path,
                    lasso_path,
                )

                linear_json = linear_path.with_suffix(
                    ".json"
                )
                lasso_json = lasso_path.with_suffix(
                    ".json"
                )
                if linear_json.is_file():
                    shutil.copy2(
                        linear_json,
                        lasso_json,
                    )

                stats = load_normalization_stats(
                    lasso_path,
                    cfg,
                )

                print(
                    "[NORMALIZATION] Reused Linear normalization"
                )
                print(
                    "        source:",
                    linear_path,
                )
                print(
                    "        Lasso :",
                    lasso_path,
                )
                return stats

    print(
        "[NORMALIZATION] Recomputing train-only normalization."
    )
    return compile_train_only_normalization(
        cfg,
        force_recompute=bool(
            RUN.NORMALIZATION_OVERWRITE
        ),
    )


# =============================================================================
# 4. Accumulate weighted sufficient statistics
# =============================================================================

def accumulate_statistics(cfg, stats):
    include_first = bool(
        cfg.BASELINE_INCLUDE_FIRST_STEP_IN_FIT
    )

    d = INPUT_DIM
    c = TARGET_DIM

    xtwx = np.zeros(
        (d, d),
        dtype=np.float64,
    )
    xtwy = np.zeros(
        (d, c),
        dtype=np.float64,
    )

    if cfg.LASSO_FIT_INTERCEPT:
        xtw1 = np.zeros(
            d,
            dtype=np.float64,
        )
        one_tw_y = np.zeros(
            c,
            dtype=np.float64,
        )
        one_tw_one = 0.0

    point_count = 0
    weight_sum = 0.0

    total_points = _selected_point_count(
        cfg,
        "train",
        include_first,
    )
    progress = Progress(
        "Lasso accumulate",
        total_points,
    )

    for chunk in iter_baseline_point_chunks(
        split_name="train",
        cfg=cfg,
        include_first_step=include_first,
        max_points=cfg.BASELINE_MAX_TRAIN_POINTS,
    ):
        x = _normalize_x(
            chunk["x"],
            stats,
            cfg,
        )
        y = _normalize_y(
            chunk["y"],
            stats,
            cfg,
        )
        w = _weights(
            chunk["weight"],
            cfg,
        )

        xtwx += (
            x.T
            @ (w[:, None] * x)
        )
        xtwy += (
            x.T
            @ (w[:, None] * y)
        )

        if cfg.LASSO_FIT_INTERCEPT:
            xtw1 += x.T @ w
            one_tw_y += w @ y
            one_tw_one += float(
                np.sum(
                    w,
                    dtype=np.float64,
                )
            )

        n = int(x.shape[0])
        point_count += n
        weight_sum += float(
            np.sum(
                w,
                dtype=np.float64,
            )
        )
        progress.update(n)

    if (
        point_count <= 0
        or weight_sum <= 0.0
    ):
        raise RuntimeError(
            "No training points accumulated"
        )

    if cfg.LASSO_FIT_INTERCEPT:
        gram = np.zeros(
            (d + 1, d + 1),
            dtype=np.float64,
        )
        rhs = np.zeros(
            (d + 1, c),
            dtype=np.float64,
        )

        gram[:d, :d] = (
            xtwx / weight_sum
        )
        gram[:d, d] = (
            xtw1 / weight_sum
        )
        gram[d, :d] = (
            xtw1 / weight_sum
        )
        gram[d, d] = (
            one_tw_one / weight_sum
        )

        rhs[:d, :] = (
            xtwy / weight_sum
        )
        rhs[d, :] = (
            one_tw_y / weight_sum
        )
    else:
        gram = (
            xtwx / weight_sum
        )
        rhs = (
            xtwy / weight_sum
        )

    return {
        "gram": gram,
        "rhs": rhs,
        "point_count": int(point_count),
        "weight_sum": float(weight_sum),
    }


# =============================================================================
# 5. Coordinate descent
# =============================================================================

def _solve_one_output_coordinate_descent(
    gram: np.ndarray,
    rhs: np.ndarray,
    *,
    alpha: float,
    fit_intercept: bool,
    max_iter: int,
    tol: float,
    selection: str,
    random_seed: int,
    output_name: str,
):
    """
    Minimize:
        beta^T G beta - 2 r^T beta
        + alpha * TARGET_DIM * sum_j |beta_j|

    over slope coordinates j=0..INPUT_DIM-1.
    Intercept, if present, is the final coordinate and is unpenalized.

    Coordinate update:
        rho_j = r_j - sum_{k != j} G_jk beta_k

        slope:
            beta_j = soft_threshold(
                rho_j,
                0.5 * alpha * TARGET_DIM
            ) / G_jj

        intercept:
            beta_b = rho_b / G_bb
    """
    gram = np.asarray(
        gram,
        dtype=np.float64,
    )
    rhs = np.asarray(
        rhs,
        dtype=np.float64,
    ).reshape(-1)

    p = int(gram.shape[0])

    if gram.shape != (p, p):
        raise ValueError(
            "Gram must be square"
        )
    if rhs.shape != (p,):
        raise ValueError(
            "rhs shape mismatch"
        )

    beta = np.zeros(
        p,
        dtype=np.float64,
    )

    slope_count = (
        p - 1
        if fit_intercept
        else p
    )

    threshold = (
        0.5
        * float(alpha)
        * float(TARGET_DIM)
    )

    rng = np.random.default_rng(
        int(random_seed)
    )

    if selection not in (
        "cyclic",
        "random",
    ):
        raise ValueError(
            "selection must be cyclic/random"
        )

    converged = False
    final_max_delta = float("inf")
    iteration = 0

    print("-" * 108)
    print(
        f"LASSO COORDINATE DESCENT | {output_name}"
    )
    print(
        "alpha             :",
        alpha,
    )
    print(
        "soft threshold    :",
        threshold,
    )
    print(
        "max iterations    :",
        max_iter,
    )
    print(
        "tolerance         :",
        tol,
    )
    print(
        "selection         :",
        selection,
    )
    print("-" * 108)

    for iteration in range(
        1,
        int(max_iter) + 1,
    ):
        old_beta = beta.copy()

        if selection == "cyclic":
            slope_order = np.arange(
                slope_count,
                dtype=np.int64,
            )
        else:
            slope_order = rng.permutation(
                slope_count
            )

        for j_raw in slope_order:
            j = int(j_raw)
            gjj = float(
                gram[j, j]
            )

            if gjj <= 0.0:
                beta[j] = 0.0
                continue

            rho = float(
                rhs[j]
                - np.dot(
                    gram[j, :],
                    beta,
                )
                + gjj * beta[j]
            )

            beta[j] = (
                _soft_threshold(
                    rho,
                    threshold,
                )
                / gjj
            )

        # Intercept is unregularized.
        if fit_intercept:
            j = p - 1
            gjj = float(
                gram[j, j]
            )
            if gjj <= 0.0:
                raise RuntimeError(
                    "Non-positive intercept Gram diagonal"
                )

            rho = float(
                rhs[j]
                - np.dot(
                    gram[j, :],
                    beta,
                )
                + gjj * beta[j]
            )
            beta[j] = rho / gjj

        delta = np.abs(
            beta - old_beta
        )
        final_max_delta = float(
            np.max(delta)
        )

        scale = max(
            1.0,
            float(
                np.max(
                    np.abs(beta)
                )
            ),
        )
        threshold_tol = (
            float(tol) * scale
        )

        if (
            iteration == 1
            or iteration
            % int(
                RUN.PRINT_EVERY_ITERATIONS
            )
            == 0
            or final_max_delta
            <= threshold_tol
            or iteration == max_iter
        ):
            nonzero = int(
                np.count_nonzero(
                    np.abs(
                        beta[:slope_count]
                    )
                    > 0.0
                )
            )
            print(
                f"iter {iteration:5d} | "
                f"max_delta={final_max_delta:.6e} | "
                f"tol_limit={threshold_tol:.6e} | "
                f"nonzero={nonzero:2d}/{slope_count}"
            )

        if (
            final_max_delta
            <= threshold_tol
        ):
            converged = True
            break

    nonzero_count = int(
        np.count_nonzero(
            np.abs(
                beta[:slope_count]
            )
            > 0.0
        )
    )

    return {
        "beta": beta,
        "iterations": int(iteration),
        "converged": bool(converged),
        "final_max_delta":
            float(final_max_delta),
        "nonzero_count":
            int(nonzero_count),
        "threshold":
            float(threshold),
    }


def fit_weighted_lasso(
    cfg,
    stats,
):
    alpha = float(
        cfg.LASSO_ALPHA
    )
    if alpha <= 0.0:
        raise ValueError(
            "LASSO_ALPHA must be positive"
        )

    accumulated = accumulate_statistics(
        cfg,
        stats,
    )
    gram = np.asarray(
        accumulated["gram"],
        dtype=np.float64,
    )
    rhs = np.asarray(
        accumulated["rhs"],
        dtype=np.float64,
    )

    fit_intercept = bool(
        cfg.LASSO_FIT_INTERCEPT
    )

    d = INPUT_DIM
    c = TARGET_DIM

    coef = np.zeros(
        (c, d),
        dtype=np.float64,
    )
    intercept = np.zeros(
        c,
        dtype=np.float64,
    )

    iteration_counts = []
    converged_flags = []
    final_deltas = []
    nonzero_counts = []

    for output_index in range(c):
        result = (
            _solve_one_output_coordinate_descent(
                gram=gram,
                rhs=rhs[:, output_index],
                alpha=alpha,
                fit_intercept=fit_intercept,
                max_iter=int(
                    cfg.LASSO_MAX_ITER
                ),
                tol=float(
                    cfg.LASSO_TOL
                ),
                selection=str(
                    cfg.LASSO_SELECTION
                ),
                random_seed=(
                    int(cfg.RANDOM_SEED)
                    + output_index
                ),
                output_name=TARGET_ORDER[
                    output_index
                ],
            )
        )

        beta = np.asarray(
            result["beta"],
            dtype=np.float64,
        )

        if fit_intercept:
            coef[
                output_index, :
            ] = beta[:d]
            intercept[
                output_index
            ] = beta[d]
        else:
            coef[
                output_index, :
            ] = beta

        iteration_counts.append(
            int(
                result["iterations"]
            )
        )
        converged_flags.append(
            bool(
                result["converged"]
            )
        )
        final_deltas.append(
            float(
                result["final_max_delta"]
            )
        )
        nonzero_counts.append(
            int(
                result["nonzero_count"]
            )
        )

    if not all(
        converged_flags
    ):
        failed = [
            TARGET_ORDER[i]
            for i, ok
            in enumerate(
                converged_flags
            )
            if not ok
        ]
        print(
            "[LASSO WARNING] Max iterations reached without "
            "the configured coefficient-change tolerance for:",
            failed,
        )

    return {
        "coef": coef,
        "intercept": intercept,
        "alpha": alpha,
        "soft_threshold":
            0.5
            * alpha
            * TARGET_DIM,
        "point_count":
            accumulated[
                "point_count"
            ],
        "weight_sum":
            accumulated[
                "weight_sum"
            ],
        "iterations":
            iteration_counts,
        "converged":
            converged_flags,
        "final_max_delta":
            final_deltas,
        "nonzero_counts":
            nonzero_counts,
        "total_nonzero":
            int(
                sum(
                    nonzero_counts
                )
            ),
    }


# =============================================================================
# 6. Evaluation
# =============================================================================

def evaluate_lasso(
    cfg,
    stats,
    fit,
    split: str,
):
    total_points = _selected_point_count(
        cfg,
        split,
        include_first=False,
    )
    progress = Progress(
        f"Eval {split}",
        total_points,
    )

    weighted_sse = 0.0
    weighted_target_sq = 0.0

    weighted_sse_comp = np.zeros(
        TARGET_DIM,
        dtype=np.float64,
    )
    weighted_target_comp = np.zeros(
        TARGET_DIM,
        dtype=np.float64,
    )

    normalized_sse = 0.0
    normalized_den = 0.0
    point_count = 0

    max_points = (
        cfg.BASELINE_MAX_TRAIN_POINTS
        if split == "train"
        else cfg.BASELINE_MAX_VAL_POINTS
    )

    coef = np.asarray(
        fit["coef"],
        dtype=np.float64,
    )
    intercept = np.asarray(
        fit["intercept"],
        dtype=np.float64,
    )

    for chunk in iter_baseline_point_chunks(
        split_name=split,
        cfg=cfg,
        include_first_step=False,
        max_points=max_points,
    ):
        x_norm = _normalize_x(
            chunk["x"],
            stats,
            cfg,
        )
        y_phys = np.asarray(
            chunk["y"],
            dtype=np.float64,
        )
        y_norm = _normalize_y(
            y_phys,
            stats,
            cfg,
        )
        w = _weights(
            chunk["weight"],
            cfg,
        )

        pred_norm = (
            x_norm @ coef.T
            + intercept[None, :]
        )
        pred_phys = _denormalize_y(
            pred_norm,
            stats,
            cfg,
        )

        err = (
            pred_phys - y_phys
        )

        weighted_sse += float(
            np.sum(
                w[:, None]
                * err
                * err,
                dtype=np.float64,
            )
        )
        weighted_target_sq += float(
            np.sum(
                w[:, None]
                * y_phys
                * y_phys,
                dtype=np.float64,
            )
        )

        weighted_sse_comp += np.sum(
            w[:, None]
            * err
            * err,
            axis=0,
            dtype=np.float64,
        )
        weighted_target_comp += np.sum(
            w[:, None]
            * y_phys
            * y_phys,
            axis=0,
            dtype=np.float64,
        )

        norm_err = (
            pred_norm - y_norm
        )
        normalized_sse += float(
            np.sum(
                w[:, None]
                * norm_err
                * norm_err,
                dtype=np.float64,
            )
        )
        normalized_den += (
            float(
                np.sum(
                    w,
                    dtype=np.float64,
                )
            )
            * TARGET_DIM
        )

        n = int(
            y_phys.shape[0]
        )
        point_count += n
        progress.update(n)

    if point_count <= 0:
        raise RuntimeError(
            f"No {split} points evaluated"
        )

    relative_l2 = (
        100.0
        * math.sqrt(
            weighted_sse
            / weighted_target_sq
        )
        if weighted_target_sq > 0.0
        else float("nan")
    )

    component = np.full(
        TARGET_DIM,
        np.nan,
        dtype=np.float64,
    )
    valid = (
        weighted_target_comp > 0.0
    )
    component[valid] = (
        100.0
        * np.sqrt(
            weighted_sse_comp[valid]
            / weighted_target_comp[valid]
        )
    )

    return {
        "split": split,
        "point_count":
            int(point_count),
        "normalized_weighted_mse":
            float(
                normalized_sse
                / normalized_den
            ),
        "weighted_global_relative_l2_pct":
            float(
                relative_l2
            ),
        "weighted_component_relative_l2_pct":
            [
                float(x)
                for x in component
            ],
        "component_names":
            list(TARGET_ORDER),
        "first_step_excluded":
            True,
    }


# =============================================================================
# 7. Save outputs
# =============================================================================

def save_outputs(
    cfg,
    model,
    fit,
    train_metrics,
    val_metrics,
    lcs,
):
    cfg.create_output_directories()
    cfg.save_snapshot()

    coef = np.asarray(
        fit["coef"],
        dtype=np.float64,
    )
    intercept = np.asarray(
        fit["intercept"],
        dtype=np.float64,
    )

    with torch.no_grad():
        model.regression_weight.copy_(
            torch.as_tensor(
                coef,
                dtype=
                    model.regression_weight.dtype,
                device=
                    model.regression_weight.device,
            )
        )

        if (
            model.regression_bias
            is not None
        ):
            model.regression_bias.copy_(
                torch.as_tensor(
                    intercept,
                    dtype=
                        model.regression_bias.dtype,
                    device=
                        model.regression_bias.device,
                )
            )

    structure_path = (
        save_model_structure(
            model,
            cfg,
        )
    )

    checkpoint = {
        "schema_name":
            "CPFE_STEP10_CLASSICAL_LASSO_V1",
        "created_time":
            _now_string(),

        "model_state_dict":
            model.state_dict(),

        "phase":
            str(cfg.PHASE),
        "multiplier":
            int(cfg.MULTIPLIER),
        "model_type":
            "lasso",
        "architecture_tag":
            str(
                cfg.MODEL_ARCHITECTURE_TAG
            ),

        "input_dim":
            INPUT_DIM,
        "target_dim":
            TARGET_DIM,

        "fit_method":
            "streamed_weighted_gram_coordinate_descent_lasso",

        "fit_intercept":
            bool(
                cfg.LASSO_FIT_INTERCEPT
            ),
        "lasso_alpha":
            float(
                fit["alpha"]
            ),
        "soft_threshold":
            float(
                fit[
                    "soft_threshold"
                ]
            ),
        "bias_regularized":
            False,

        "lasso_max_iter":
            int(
                cfg.LASSO_MAX_ITER
            ),
        "lasso_tol":
            float(
                cfg.LASSO_TOL
            ),
        "lasso_selection":
            str(
                cfg.LASSO_SELECTION
            ),

        "fit_point_count":
            int(
                fit["point_count"]
            ),
        "fit_weight_sum":
            float(
                fit["weight_sum"]
            ),

        "iterations_per_component":
            list(
                fit["iterations"]
            ),
        "converged_per_component":
            list(
                fit["converged"]
            ),
        "final_max_delta_per_component":
            list(
                fit[
                    "final_max_delta"
                ]
            ),
        "nonzero_per_component":
            list(
                fit[
                    "nonzero_counts"
                ]
            ),
        "total_nonzero":
            int(
                fit[
                    "total_nonzero"
                ]
            ),

        "normalization_path":
            str(
                Path(
                    cfg.NORMALIZATION_PATH
                )
            ),

        "train_metrics":
            train_metrics,
        "val_metrics":
            val_metrics,
        "lcs_line_pct":
            float(
                lcs.horizontal_line_pct
            ),

        "config_snapshot":
            cfg.snapshot_dict(),
    }

    for raw in (
        cfg.BEST_CHECKPOINT_PATH,
        cfg.LAST_CHECKPOINT_PATH,
    ):
        path = Path(raw)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp = path.with_suffix(
            path.suffix + ".tmp"
        )
        torch.save(
            checkpoint,
            temp,
        )
        os.replace(
            temp,
            path,
        )

    val_error = float(
        val_metrics[
            "weighted_global_relative_l2_pct"
        ]
    )
    lcs_error = float(
        lcs.horizontal_line_pct
    )

    summary = {
        "schema_name":
            "CPFE_STEP10_CLASSICAL_LASSO_SUMMARY_V1",
        "created_time":
            _now_string(),

        "phase":
            str(cfg.PHASE),
        "multiplier":
            int(cfg.MULTIPLIER),
        "model_type":
            "lasso",

        "fit_method":
            "streamed_weighted_gram_coordinate_descent_lasso",

        "uses_optimizer":
            False,
        "uses_epochs":
            False,
        "uses_sequence_history":
            False,
        "uses_coordinate_descent":
            True,

        "lasso_alpha":
            float(
                fit["alpha"]
            ),
        "soft_threshold":
            float(
                fit[
                    "soft_threshold"
                ]
            ),
        "fit_intercept":
            bool(
                cfg.LASSO_FIT_INTERCEPT
            ),
        "bias_regularized":
            False,

        "lasso_max_iter":
            int(
                cfg.LASSO_MAX_ITER
            ),
        "lasso_tol":
            float(
                cfg.LASSO_TOL
            ),
        "lasso_selection":
            str(
                cfg.LASSO_SELECTION
            ),

        "fit_point_count":
            int(
                fit["point_count"]
            ),

        "iterations_per_component":
            list(
                fit["iterations"]
            ),
        "converged_per_component":
            list(
                fit["converged"]
            ),
        "nonzero_per_component":
            list(
                fit[
                    "nonzero_counts"
                ]
            ),
        "total_nonzero":
            int(
                fit[
                    "total_nonzero"
                ]
            ),
        "total_possible_weights":
            int(
                INPUT_DIM
                * TARGET_DIM
            ),

        "train":
            train_metrics,
        "val":
            val_metrics,

        "lcs_line_pct":
            lcs_error,
        "val_minus_lcs_pct_point":
            val_error
            - lcs_error,
        "below_lcs":
            val_error
            < lcs_error,

        "best_checkpoint":
            str(
                cfg.BEST_CHECKPOINT_PATH
            ),
        "normalization":
            str(
                cfg.NORMALIZATION_PATH
            ),
        "model_structure":
            str(
                structure_path
            ),
    }

    Path(
        cfg.TRAINING_SUMMARY_PATH
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = (
        Path(
            cfg.EXPERIMENT_ROOT
        )
        / "classical_fit_result.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        fieldnames = (
            "phase",
            "multiplier",
            "model",
            "lasso_alpha",
            "train_error_pct",
            "val_error_pct",
            "lcs_pct",
            "val_minus_lcs_pct_point",
            "below_lcs",
            "fit_points",
            "sigma11_iter",
            "sigma22_iter",
            "sigma33_iter",
            "sigma12_iter",
            "sigma13_iter",
            "sigma23_iter",
            "sigma11_nonzero",
            "sigma22_nonzero",
            "sigma33_nonzero",
            "sigma12_nonzero",
            "sigma13_nonzero",
            "sigma23_nonzero",
            "total_nonzero",
        )

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        row = {
            "phase":
                cfg.PHASE,
            "multiplier":
                cfg.MULTIPLIER,
            "model":
                "lasso",
            "lasso_alpha":
                fit["alpha"],
            "train_error_pct":
                train_metrics[
                    "weighted_global_relative_l2_pct"
                ],
            "val_error_pct":
                val_error,
            "lcs_pct":
                lcs_error,
            "val_minus_lcs_pct_point":
                val_error
                - lcs_error,
            "below_lcs":
                val_error
                < lcs_error,
            "fit_points":
                fit[
                    "point_count"
                ],
            "total_nonzero":
                fit[
                    "total_nonzero"
                ],
        }

        for i, name in enumerate(
            TARGET_ORDER
        ):
            row[
                f"{name}_iter"
            ] = fit[
                "iterations"
            ][i]
            row[
                f"{name}_nonzero"
            ] = fit[
                "nonzero_counts"
            ][i]

        writer.writerow(row)

    # Lasso has coordinate-descent iterations but no epoch history.
    if (
        RUN.ALLOW_OVERWRITE_EXISTING_RESULTS
    ):
        old_history = Path(
            cfg.HISTORY_CSV_PATH
        )
        if old_history.exists():
            old_history.unlink()

    print("=" * 108)
    print("CORRECT CLASSICAL LASSO COMPLETE")
    print("Phase             :", cfg.PHASE)
    print("Multiplier        :", cfg.MULTIPLIER)
    print("LASSO_ALPHA       :", fit["alpha"])
    print("Soft threshold    :", fit["soft_threshold"])
    print("Fit points        :", f"{fit['point_count']:,}")
    print("Iterations        :", fit["iterations"])
    print("Converged         :", fit["converged"])
    print("Nonzero/component :", fit["nonzero_counts"])
    print(
        "Total nonzero     :",
        f"{fit['total_nonzero']}/{INPUT_DIM * TARGET_DIM}",
    )
    print(
        "Train error       : %.6f %%"
        % train_metrics[
            "weighted_global_relative_l2_pct"
        ]
    )
    print(
        "Validation error  : %.6f %%"
        % val_error
    )
    print(
        "LCS line          : %.6f %%"
        % lcs_error
    )
    print(
        "Val - LCS         : %.6f pct-point"
        % (
            val_error
            - lcs_error
        )
    )
    print(
        "Below LCS         :",
        "YES"
        if val_error < lcs_error
        else "NO",
    )
    print(
        "Best checkpoint   :",
        cfg.BEST_CHECKPOINT_PATH,
    )
    print(
        "Summary           :",
        cfg.TRAINING_SUMMARY_PATH,
    )
    print(
        "Classical CSV     :",
        csv_path,
    )
    print(
        "Epoch history     : NONE "
        "(coordinate-descent iterations are reported separately)"
    )
    print("=" * 108)


# =============================================================================
# 8. One phase
# =============================================================================

def run_phase(
    phase: str,
) -> None:
    overrides = {
        "PHASE":
            str(
                phase
            ).upper(),
        "MODEL_TYPE":
            "lasso",
        "MULTIPLIER":
            int(
                RUN.MULTIPLIER
            ),
        "DEVICE":
            str(
                RUN.DEVICE
            ),
        "DATA_ACCESS_MODE":
            "h5",
        "NORMALIZATION_OVERWRITE":
            bool(
                RUN.NORMALIZATION_OVERWRITE
            ),
        "ENABLE_LIVE_DASHBOARD":
            False,
        "DASHBOARD_SHOW_WINDOW":
            False,
        "RESUME_MODE":
            "never",
    }

    if (
        RUN.LASSO_ALPHA_OVERRIDE
        is not None
    ):
        overrides[
            "LASSO_ALPHA"
        ] = float(
            RUN.LASSO_ALPHA_OVERRIDE
        )

    cfg = dataclasses.replace(
        BASE_CFG,
        **overrides,
    )

    cfg.validate()
    cfg.validate_required_files()

    info = inspect_phase_h5(
        cfg
    )
    lcs = load_lcs_reference(
        cfg
    )

    print("=" * 108)
    print("Correct classical Lasso Regression")
    print("Phase             :", cfg.PHASE)
    print("Multiplier        :", cfg.MULTIPLIER)
    print("Sequence length   :", info.sequence_length)
    print("Train trajectories:", f"{info.train_count:,}")
    print("Val trajectories  :", f"{info.val_count:,}")
    print("LASSO_ALPHA       :", cfg.LASSO_ALPHA)
    print("Fit intercept     :", cfg.LASSO_FIT_INTERCEPT)
    print("Fit first step    :", cfg.BASELINE_INCLUDE_FIRST_STEP_IN_FIT)
    print("Max iterations    :", cfg.LASSO_MAX_ITER)
    print("Tolerance         :", cfg.LASSO_TOL)
    print("Selection         :", cfg.LASSO_SELECTION)
    print("Optimizer         : NONE")
    print("Epochs            : NONE")
    print("Solve             : weighted coordinate-descent Lasso")
    print("Output root       :", cfg.EXPERIMENT_ROOT)
    print("=" * 108)

    _check_clean_output(
        cfg
    )

    if RUN.CHECK_ONLY:
        print(
            "[CHECK ONLY] Paths, H5, LCS and Lasso settings are valid."
        )
        return

    stats = _prepare_normalization(
        cfg
    )

    fit = fit_weighted_lasso(
        cfg,
        stats,
    )

    train_metrics = evaluate_lasso(
        cfg,
        stats,
        fit,
        split="train",
    )

    val_metrics = evaluate_lasso(
        cfg,
        stats,
        fit,
        split="val",
    )

    model = build_model(
        cfg
    )

    save_outputs(
        cfg=cfg,
        model=model,
        fit=fit,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        lcs=lcs,
    )


# =============================================================================
# 9. Main
# =============================================================================

def main() -> int:
    if (
        RUN.LASSO_ALPHA_OVERRIDE
        is not None
        and RUN.LASSO_ALPHA_OVERRIDE
        <= 0.0
    ):
        raise ValueError(
            "LASSO_ALPHA_OVERRIDE must be positive"
        )

    if (
        int(
            RUN.PRINT_EVERY_ITERATIONS
        )
        <= 0
    ):
        raise ValueError(
            "PRINT_EVERY_ITERATIONS must be positive"
        )

    for phase in RUN.PHASES:
        run_phase(
            phase
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
