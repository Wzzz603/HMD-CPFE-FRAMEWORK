#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_9_train_polynomial2_bcc_hcp_FIXED.py

Correct second-order Polynomial Regression baseline for Step-10.

Core idea
---------
Original normalized current-step input:
    x in R^22

Second-order polynomial expansion (include_bias=False):
    [x1, ..., x22,
     x1^2, x1*x2, ..., x21*x22, x22^2]

Feature dimension:
    22 + 22*23/2 = 275

Regression:
    y_hat = W phi(x) + b
    W: [6, 275]
    b: [6]

Fit objective
-------------
Same weighted data objective used by the Step-10 baseline family:

    argmin sum_i w_i || y_i - (W phi(x_i) + b) ||^2

No L1/L2 regularization is added in this Polynomial-2 baseline.
Therefore this isolates the effect of explicit second-order nonlinearity.

Important
---------
- This script DOES NOT call 10_7_train.py or 10_6_trainer.py.
- There are NO Adam/AdamW epochs.
- It performs one streamed weighted OLS fit.
- Time history is NOT used; each time point is an independent regression point.
- BCC and HCP are fitted separately.
- Step-08 train/val split is reused.
- Train-only normalization is reused from the compatible Linear experiment.
- The first time point may be included in fitting, but Train/Val metrics used
  against LCS exclude time index 0, matching the existing Step-09 convention.

Performance
-----------
A 275-feature Gram matrix is much more expensive to accumulate than the
22-feature Linear/Ridge/Lasso baselines. This script therefore:
- reads H5 point chunks on CPU;
- expands and accumulates polynomial sufficient statistics on CUDA when
  available;
- uses accurate float32 GEMM with TF32 disabled;
- accumulates each subchunk's Gram/RHS into CPU float64 arrays;
- solves only the final 276 x 276 system on CPU in float64.

Outputs
-------
For each phase:
    <PROJECT_ROOT>/10_training/
      <DATASET_RUN_NAME>__<PHASE>__mXX__polynomial2__poly2_275/

        checkpoints/best_checkpoint.pt
        checkpoints/last_checkpoint.pt
        normalization_train_only.npz
        config_snapshot.json
        model_structure.json
        polynomial_feature_map.json
        training_summary.json
        classical_fit_result.csv

The checkpoint stores:
    model_state_dict["linear.weight"] : [6,275]
    model_state_dict["linear.bias"]   : [6]
plus the polynomial feature definition needed by a later Step-12 predictor.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn


# =============================================================================
# 0. USER CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class LauncherConfig:
    PHASES: Tuple[str, ...] = ("BCC", "HCP")
    MULTIPLIER: int = 1

    # "auto" -> CUDA when available, otherwise CPU.
    DEVICE: str = "auto"

    # Internal point subchunk used for polynomial expansion / GEMM.
    # 200k x 275 float32 is about 210 MiB before temporary tensors.
    POLY_SUBCHUNK_POINTS: int = 200_000

    # Disable TF32 so the streamed sufficient statistics are accumulated from
    # ordinary float32 GEMMs, then promoted to CPU float64 between subchunks.
    CUDA_ALLOW_TF32: bool = False

    # Prefer the already validated Linear normalization for the same H5.
    REUSE_LINEAR_NORMALIZATION: bool = True

    # Safety against mixing incompatible previous outputs.
    ALLOW_OVERWRITE_EXISTING_RESULTS: bool = False

    # Normally False. Set True only if the Step-08 H5 itself changed.
    NORMALIZATION_OVERWRITE: bool = False

    # Validate only; do not fit.
    CHECK_ONLY: bool = False


RUN = LauncherConfig()


# =============================================================================
# 1. Local Step-10 imports
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


_cfg_mod = _import_local(
    "step10_cfg_for_poly2",
    "10_1_config.py",
)
_data_mod = _import_local(
    "step10_data_for_poly2",
    "10_2_h5_dataset.py",
)
_norm_mod = _import_local(
    "step10_norm_for_poly2",
    "10_3_normalizer.py",
)

BASE_CFG = _cfg_mod.CFG
INPUT_DIM = int(_cfg_mod.INPUT_DIM)
TARGET_DIM = int(_cfg_mod.TARGET_DIM)
INPUT_ORDER = tuple(_cfg_mod.INPUT_ORDER)
TARGET_ORDER = tuple(_cfg_mod.TARGET_ORDER)

inspect_phase_h5 = _data_mod.inspect_phase_h5
load_lcs_reference = _data_mod.load_lcs_reference
iter_baseline_point_chunks = _data_mod.iter_baseline_point_chunks

compile_train_only_normalization = (
    _norm_mod.compile_train_only_normalization
)
load_normalization_stats = (
    _norm_mod.load_normalization_stats
)


# =============================================================================
# 2. Polynomial feature definition
# =============================================================================

PAIR_I = np.asarray(
    [
        i
        for i in range(INPUT_DIM)
        for j in range(i, INPUT_DIM)
    ],
    dtype=np.int64,
)
PAIR_J = np.asarray(
    [
        j
        for i in range(INPUT_DIM)
        for j in range(i, INPUT_DIM)
    ],
    dtype=np.int64,
)

SECOND_ORDER_DIM = int(PAIR_I.size)
POLY_DIM = int(INPUT_DIM + SECOND_ORDER_DIM)

if SECOND_ORDER_DIM != INPUT_DIM * (INPUT_DIM + 1) // 2:
    raise RuntimeError("Polynomial pair count is inconsistent")
if POLY_DIM != 275:
    raise RuntimeError(
        f"Expected 275 second-order polynomial features, got {POLY_DIM}"
    )


def _polynomial_feature_names() -> Tuple[str, ...]:
    names = list(INPUT_ORDER)

    for i, j in zip(
        PAIR_I.tolist(),
        PAIR_J.tolist(),
    ):
        if i == j:
            names.append(
                f"{INPUT_ORDER[i]}^2"
            )
        else:
            names.append(
                f"{INPUT_ORDER[i]}*{INPUT_ORDER[j]}"
            )

    if len(names) != POLY_DIM:
        raise RuntimeError("Polynomial feature-name count mismatch")

    return tuple(names)


POLY_FEATURE_NAMES = _polynomial_feature_names()


# =============================================================================
# 3. Output paths
# =============================================================================

@dataclass(frozen=True)
class OutputPaths:
    root: Path
    checkpoint_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    normalization: Path
    config_snapshot: Path
    structure_json: Path
    feature_map_json: Path
    summary_json: Path
    result_csv: Path


def _build_output_paths(cfg) -> OutputPaths:
    root = (
        Path(cfg.PROJECT_ROOT)
        / cfg.STEP10_ROOT_NAME
        / (
            f"{cfg.DATASET_RUN_NAME}"
            f"__{cfg.PHASE}"
            f"__m{cfg.MULTIPLIER:02d}"
            f"__polynomial2"
            f"__poly2_{POLY_DIM}"
        )
    )

    checkpoint_dir = root / "checkpoints"

    return OutputPaths(
        root=root,
        checkpoint_dir=checkpoint_dir,
        best_checkpoint=checkpoint_dir / "best_checkpoint.pt",
        last_checkpoint=checkpoint_dir / "last_checkpoint.pt",
        normalization=root / "normalization_train_only.npz",
        config_snapshot=root / "config_snapshot.json",
        structure_json=root / "model_structure.json",
        feature_map_json=root / "polynomial_feature_map.json",
        summary_json=root / "training_summary.json",
        result_csv=root / "classical_fit_result.csv",
    )


# =============================================================================
# 4. Utilities
# =============================================================================

def _now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _resolve_device() -> torch.device:
    raw = str(RUN.DEVICE).strip().lower()

    if raw == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(raw)


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
        filled = min(
            width,
            int(round(width * ratio)),
        )
        bar = (
            "#" * filled
            + "-" * (width - filled)
        )

        elapsed = (
            time.perf_counter()
            - self.started
        )
        rate = (
            self.current
            / max(elapsed, 1.0e-12)
        )
        remaining = (
            (self.total - self.current) / rate
            if rate > 0.0
            else float("inf")
        )

        sys.stdout.write(
            f"\r{self.label:<16s} "
            f"[{bar}] "
            f"{100.0 * ratio:6.2f}% "
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
        ntraj = min(
            ntraj,
            int(max_traj),
        )

    ntime = int(info.sequence_length)
    if not include_first:
        ntime -= 1

    total = int(
        ntraj * ntime
    )

    if max_points is not None:
        total = min(
            total,
            int(max_points),
        )

    return total


def _normalize_x(
    x: np.ndarray,
    stats,
    cfg,
) -> np.ndarray:
    x = np.asarray(
        x,
        dtype=np.float32,
    )

    if not cfg.NORMALIZE_INPUTS:
        return x

    return np.asarray(
        (
            x
            - stats.input_mean.astype(
                np.float32
            )[None, :]
        )
        / stats.input_std.astype(
            np.float32
        )[None, :],
        dtype=np.float32,
    )


def _normalize_y(
    y: np.ndarray,
    stats,
    cfg,
) -> np.ndarray:
    y = np.asarray(
        y,
        dtype=np.float32,
    )

    if not cfg.NORMALIZE_TARGETS:
        return y

    return np.asarray(
        (
            y
            - stats.target_mean.astype(
                np.float32
            )[None, :]
        )
        / stats.target_std.astype(
            np.float32
        )[None, :],
        dtype=np.float32,
    )


def _denormalize_y_numpy(
    y_norm: np.ndarray,
    stats,
    cfg,
) -> np.ndarray:
    y_norm = np.asarray(
        y_norm,
        dtype=np.float64,
    )

    if not cfg.NORMALIZE_TARGETS:
        return y_norm

    return (
        y_norm
        * np.asarray(
            stats.target_std,
            dtype=np.float64,
        )[None, :]
        + np.asarray(
            stats.target_mean,
            dtype=np.float64,
        )[None, :]
    )


def _weights(
    raw: np.ndarray,
    cfg,
) -> np.ndarray:
    raw = np.asarray(
        raw,
        dtype=np.float32,
    ).reshape(-1)

    if cfg.USE_SAMPLE_WEIGHT:
        w = raw
    else:
        w = np.ones_like(
            raw,
            dtype=np.float32,
        )

    if not np.isfinite(w).all():
        raise ValueError(
            "Sample weights contain NaN/Inf"
        )
    if np.any(w <= 0.0):
        raise ValueError(
            "Sample weights must be positive"
        )

    return w


def _expand_poly2_torch(
    x: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be [P,22]")
    if x.shape[1] != INPUT_DIM:
        raise ValueError(
            f"x has {x.shape[1]} inputs, expected {INPUT_DIM}"
        )

    second = (
        x[:, pair_i]
        * x[:, pair_j]
    )

    z = torch.cat(
        (x, second),
        dim=1,
    )

    if z.shape[1] != POLY_DIM:
        raise RuntimeError(
            f"Polynomial expansion has {z.shape[1]} features, expected {POLY_DIM}"
        )

    return z


def _check_clean_output(
    paths: OutputPaths,
) -> None:
    conflicts = (
        paths.best_checkpoint,
        paths.last_checkpoint,
        paths.summary_json,
        paths.result_csv,
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
            "Existing Polynomial-2 results were found.\n"
            "Rename/remove the existing experiment folder first, or set "
            "ALLOW_OVERWRITE_EXISTING_RESULTS=True.\n"
            + "\n".join(
                f"  - {path}"
                for path in existing
            )
        )


# =============================================================================
# 5. Normalization
# =============================================================================

def _prepare_normalization(
    cfg,
    paths: OutputPaths,
):
    output_norm = paths.normalization

    # First reuse a normalization already copied into the Polynomial folder.
    if (
        output_norm.is_file()
        and not RUN.NORMALIZATION_OVERWRITE
    ):
        try:
            # load_normalization_stats validates the source H5/split.  The cfg
            # passed here is a Linear data cfg, but normalization is model
            # independent and paths are supplied explicitly.
            stats = load_normalization_stats(
                output_norm,
                cfg,
            )
        except Exception as exc:
            print(
                "[NORMALIZATION] Existing Polynomial file incompatible:",
                exc,
            )
        else:
            print(
                "[NORMALIZATION] Reusing Polynomial normalization:",
                output_norm,
            )
            return stats

    # The current cfg intentionally has MODEL_TYPE="linear", so its native
    # normalization path points at the correct Linear experiment.
    linear_norm = Path(
        cfg.NORMALIZATION_PATH
    )

    if (
        RUN.REUSE_LINEAR_NORMALIZATION
        and linear_norm.is_file()
        and not RUN.NORMALIZATION_OVERWRITE
    ):
        stats = load_normalization_stats(
            linear_norm,
            cfg,
        )

        output_norm.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.copy2(
            linear_norm,
            output_norm,
        )

        linear_json = (
            linear_norm.with_suffix(".json")
        )
        output_json = (
            output_norm.with_suffix(".json")
        )
        if linear_json.is_file():
            shutil.copy2(
                linear_json,
                output_json,
            )

        copied = load_normalization_stats(
            output_norm,
            cfg,
        )

        print(
            "[NORMALIZATION] Reused compatible Linear normalization"
        )
        print(
            "        source     :",
            linear_norm,
        )
        print(
            "        Polynomial :",
            output_norm,
        )
        return copied

    print(
        "[NORMALIZATION] No reusable Linear normalization found; "
        "compiling train-only normalization first."
    )

    # This creates the standard Linear normalization in its own Linear folder.
    stats = compile_train_only_normalization(
        cfg,
        force_recompute=bool(
            RUN.NORMALIZATION_OVERWRITE
        ),
    )

    output_norm.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copy2(
        Path(cfg.NORMALIZATION_PATH),
        output_norm,
    )

    linear_json = Path(
        cfg.NORMALIZATION_PATH
    ).with_suffix(".json")
    output_json = (
        output_norm.with_suffix(".json")
    )
    if linear_json.is_file():
        shutil.copy2(
            linear_json,
            output_json,
        )

    return load_normalization_stats(
        output_norm,
        cfg,
    )


# =============================================================================
# 6. Streamed weighted Polynomial-2 OLS
# =============================================================================

def fit_weighted_poly2(
    cfg,
    stats,
    device: torch.device,
):
    include_first = bool(
        cfg.BASELINE_INCLUDE_FIRST_STEP_IN_FIT
    )

    augmented_dim = (
        POLY_DIM + 1
    )  # +1 intercept

    gram = np.zeros(
        (
            augmented_dim,
            augmented_dim,
        ),
        dtype=np.float64,
    )
    rhs = np.zeros(
        (
            augmented_dim,
            TARGET_DIM,
        ),
        dtype=np.float64,
    )

    pair_i = torch.as_tensor(
        PAIR_I,
        dtype=torch.long,
        device=device,
    )
    pair_j = torch.as_tensor(
        PAIR_J,
        dtype=torch.long,
        device=device,
    )

    total_points = _selected_point_count(
        cfg,
        split="train",
        include_first=include_first,
    )
    progress = Progress(
        "Poly2 accumulate",
        total_points,
    )

    point_count = 0
    weight_sum = 0.0

    subchunk = int(
        RUN.POLY_SUBCHUNK_POINTS
    )
    if subchunk <= 0:
        raise ValueError(
            "POLY_SUBCHUNK_POINTS must be positive"
        )

    for chunk in iter_baseline_point_chunks(
        split_name="train",
        cfg=cfg,
        include_first_step=include_first,
        max_points=cfg.BASELINE_MAX_TRAIN_POINTS,
    ):
        x_all = np.asarray(
            chunk["x"],
            dtype=np.float32,
        )
        y_all = np.asarray(
            chunk["y"],
            dtype=np.float32,
        )
        w_all = _weights(
            chunk["weight"],
            cfg,
        )

        for start in range(
            0,
            x_all.shape[0],
            subchunk,
        ):
            end = min(
                start + subchunk,
                x_all.shape[0],
            )

            x_np = _normalize_x(
                x_all[start:end],
                stats,
                cfg,
            )
            y_np = _normalize_y(
                y_all[start:end],
                stats,
                cfg,
            )
            w_np = np.asarray(
                w_all[start:end],
                dtype=np.float32,
            )

            x = torch.as_tensor(
                x_np,
                dtype=torch.float32,
                device=device,
            )
            y = torch.as_tensor(
                y_np,
                dtype=torch.float32,
                device=device,
            )
            w = torch.as_tensor(
                w_np,
                dtype=torch.float32,
                device=device,
            )

            z = _expand_poly2_torch(
                x,
                pair_i,
                pair_j,
            )

            ones = torch.ones(
                (
                    z.shape[0],
                    1,
                ),
                dtype=torch.float32,
                device=device,
            )
            z_aug = torch.cat(
                (z, ones),
                dim=1,
            )

            sqrt_w = torch.sqrt(
                w
            ).unsqueeze(1)

            zw = (
                z_aug
                * sqrt_w
            )
            yw = (
                y
                * sqrt_w
            )

            gram_chunk = (
                zw.T @ zw
            )
            rhs_chunk = (
                zw.T @ yw
            )

            gram += (
                gram_chunk
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=False,
                )
            )
            rhs += (
                rhs_chunk
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=False,
                )
            )

            n = int(
                end - start
            )
            point_count += n
            weight_sum += float(
                np.sum(
                    w_np,
                    dtype=np.float64,
                )
            )

            progress.update(n)

            del (
                x,
                y,
                w,
                z,
                ones,
                z_aug,
                sqrt_w,
                zw,
                yw,
                gram_chunk,
                rhs_chunk,
            )

        del (
            x_all,
            y_all,
            w_all,
        )

    if (
        point_count <= 0
        or weight_sum <= 0.0
    ):
        raise RuntimeError(
            "No training points accumulated"
        )

    # Common scaling does not change the OLS minimizer.
    gram /= weight_sum
    rhs /= weight_sum

    print(
        "[POLY2 SOLVE] Solving 276 x 276 weighted OLS system in float64 ..."
    )

    beta, _, rank, singular_values = (
        np.linalg.lstsq(
            gram,
            rhs,
            rcond=None,
        )
    )

    coef = (
        beta[:POLY_DIM, :].T
    )
    intercept = (
        beta[POLY_DIM, :]
    )

    if coef.shape != (
        TARGET_DIM,
        POLY_DIM,
    ):
        raise RuntimeError(
            f"Invalid coefficient shape {coef.shape}"
        )

    condition_number = (
        float(
            singular_values[0]
            / singular_values[-1]
        )
        if (
            singular_values.size
            and singular_values[-1] > 0.0
        )
        else float("inf")
    )

    return {
        "coef": coef,
        "intercept": intercept,
        "point_count":
            int(point_count),
        "weight_sum":
            float(weight_sum),
        "rank":
            int(rank),
        "condition_number":
            condition_number,
        "singular_values":
            singular_values,
    }


# =============================================================================
# 7. Evaluation
# =============================================================================

def evaluate_poly2(
    cfg,
    stats,
    fit,
    split: str,
    device: torch.device,
):
    total_points = _selected_point_count(
        cfg,
        split=split,
        include_first=False,
    )
    progress = Progress(
        f"Eval {split}",
        total_points,
    )

    coef = torch.as_tensor(
        np.asarray(
            fit["coef"],
            dtype=np.float32,
        ),
        dtype=torch.float32,
        device=device,
    )
    intercept = torch.as_tensor(
        np.asarray(
            fit["intercept"],
            dtype=np.float32,
        ),
        dtype=torch.float32,
        device=device,
    )

    target_mean = torch.as_tensor(
        np.asarray(
            stats.target_mean,
            dtype=np.float32,
        ),
        dtype=torch.float32,
        device=device,
    )
    target_std = torch.as_tensor(
        np.asarray(
            stats.target_std,
            dtype=np.float32,
        ),
        dtype=torch.float32,
        device=device,
    )

    pair_i = torch.as_tensor(
        PAIR_I,
        dtype=torch.long,
        device=device,
    )
    pair_j = torch.as_tensor(
        PAIR_J,
        dtype=torch.long,
        device=device,
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

    subchunk = int(
        RUN.POLY_SUBCHUNK_POINTS
    )

    for chunk in iter_baseline_point_chunks(
        split_name=split,
        cfg=cfg,
        include_first_step=False,
        max_points=max_points,
    ):
        x_all = np.asarray(
            chunk["x"],
            dtype=np.float32,
        )
        y_all = np.asarray(
            chunk["y"],
            dtype=np.float32,
        )
        w_all = _weights(
            chunk["weight"],
            cfg,
        )

        for start in range(
            0,
            x_all.shape[0],
            subchunk,
        ):
            end = min(
                start + subchunk,
                x_all.shape[0],
            )

            x_np = _normalize_x(
                x_all[start:end],
                stats,
                cfg,
            )
            y_phys_np = np.asarray(
                y_all[start:end],
                dtype=np.float32,
            )
            y_norm_np = _normalize_y(
                y_phys_np,
                stats,
                cfg,
            )
            w_np = np.asarray(
                w_all[start:end],
                dtype=np.float32,
            )

            x = torch.as_tensor(
                x_np,
                dtype=torch.float32,
                device=device,
            )
            y_phys = torch.as_tensor(
                y_phys_np,
                dtype=torch.float32,
                device=device,
            )
            y_norm = torch.as_tensor(
                y_norm_np,
                dtype=torch.float32,
                device=device,
            )
            w = torch.as_tensor(
                w_np,
                dtype=torch.float32,
                device=device,
            )

            z = _expand_poly2_torch(
                x,
                pair_i,
                pair_j,
            )

            pred_norm = (
                z @ coef.T
                + intercept[None, :]
            )

            if cfg.NORMALIZE_TARGETS:
                pred_phys = (
                    pred_norm
                    * target_std[None, :]
                    + target_mean[None, :]
                )
            else:
                pred_phys = pred_norm

            err = (
                pred_phys
                - y_phys
            )
            squared_error = (
                err * err
            )
            squared_target = (
                y_phys
                * y_phys
            )

            weighted_sse += float(
                torch.sum(
                    w[:, None]
                    * squared_error
                ).item()
            )
            weighted_target_sq += float(
                torch.sum(
                    w[:, None]
                    * squared_target
                ).item()
            )

            weighted_sse_comp += (
                torch.sum(
                    w[:, None]
                    * squared_error,
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=False,
                )
            )
            weighted_target_comp += (
                torch.sum(
                    w[:, None]
                    * squared_target,
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=False,
                )
            )

            norm_err = (
                pred_norm
                - y_norm
            )
            normalized_sse += float(
                torch.sum(
                    w[:, None]
                    * norm_err
                    * norm_err
                ).item()
            )
            normalized_den += (
                float(
                    torch.sum(w).item()
                )
                * TARGET_DIM
            )

            n = int(
                end - start
            )
            point_count += n
            progress.update(n)

            del (
                x,
                y_phys,
                y_norm,
                w,
                z,
                pred_norm,
                pred_phys,
                err,
                squared_error,
                squared_target,
                norm_err,
            )

        del (
            x_all,
            y_all,
            w_all,
        )

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
        "split":
            split,
        "point_count":
            int(point_count),
        "normalized_weighted_mse":
            float(
                normalized_sse
                / normalized_den
            ),
        "weighted_global_relative_l2_pct":
            float(relative_l2),
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
# 8. Save artifacts
# =============================================================================

class Polynomial2Model(nn.Module):
    """
    Deployment/checkpoint container only.

    Input here is already phi(x) with 275 polynomial features.
    """
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(
            POLY_DIM,
            TARGET_DIM,
            bias=True,
        )

    def forward(
        self,
        poly_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(
            poly_features
        )


def _save_json(
    path: Path,
    payload,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp = path.with_suffix(
        path.suffix + ".tmp"
    )
    temp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(
        temp,
        path,
    )


def save_outputs(
    cfg,
    paths: OutputPaths,
    model: Polynomial2Model,
    fit,
    train_metrics,
    val_metrics,
    lcs,
    device: torch.device,
):
    paths.root.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths.checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    coef = np.asarray(
        fit["coef"],
        dtype=np.float64,
    )
    intercept = np.asarray(
        fit["intercept"],
        dtype=np.float64,
    )

    with torch.no_grad():
        model.linear.weight.copy_(
            torch.as_tensor(
                coef,
                dtype=model.linear.weight.dtype,
                device=model.linear.weight.device,
            )
        )
        model.linear.bias.copy_(
            torch.as_tensor(
                intercept,
                dtype=model.linear.bias.dtype,
                device=model.linear.bias.device,
            )
        )

    feature_map = {
        "schema_name":
            "CPFE_STEP10_POLYNOMIAL_FEATURE_MAP_V1",
        "degree":
            2,
        "include_bias_feature":
            False,
        "original_input_dim":
            INPUT_DIM,
        "polynomial_feature_dim":
            POLY_DIM,
        "original_input_order":
            list(INPUT_ORDER),
        "feature_order":
            list(POLY_FEATURE_NAMES),
        "second_order_pair_i":
            [
                int(x)
                for x in PAIR_I
            ],
        "second_order_pair_j":
            [
                int(x)
                for x in PAIR_J
            ],
        "expansion_rule":
            "phi=[x_1..x_22, x_i*x_j for 0<=i<=j<22]",
        "expansion_applied_after_input_normalization":
            True,
    }
    _save_json(
        paths.feature_map_json,
        feature_map,
    )

    structure = {
        "schema_name":
            "CPFE_STEP10_POLYNOMIAL2_STRUCTURE_V1",
        "model_type":
            "polynomial2",
        "display_name":
            "Second-order Polynomial Regression",
        "degree":
            2,
        "uses_sequence_history":
            False,
        "uses_explicit_nonlinear_features":
            True,
        "regularization":
            "none",
        "input_dim_before_expansion":
            INPUT_DIM,
        "input_dim_after_expansion":
            POLY_DIM,
        "target_dim":
            TARGET_DIM,
        "coefficient_count":
            int(
                POLY_DIM * TARGET_DIM
                + TARGET_DIM
            ),
        "phase":
            str(cfg.PHASE),
        "multiplier":
            int(cfg.MULTIPLIER),
        "feature_map":
            str(paths.feature_map_json),
    }
    _save_json(
        paths.structure_json,
        structure,
    )

    snapshot = {
        "schema_name":
            "CPFE_STEP10_POLYNOMIAL2_CONFIG_V1",
        "created_time":
            _now_string(),
        "phase":
            str(cfg.PHASE),
        "multiplier":
            int(cfg.MULTIPLIER),
        "dataset_run_name":
            str(cfg.DATASET_RUN_NAME),
        "source_h5":
            str(cfg.H5_DATA_PATH),
        "source_lcs":
            str(cfg.LCS_BASELINE_H5_PATH),
        "source_linear_normalization":
            str(cfg.NORMALIZATION_PATH),
        "copied_polynomial_normalization":
            str(paths.normalization),
        "degree":
            2,
        "poly_dim":
            POLY_DIM,
        "fit_method":
            "streamed_weighted_OLS_on_degree2_polynomial_features",
        "regularization":
            "none",
        "device_used_for_accumulation":
            str(device),
        "poly_subchunk_points":
            int(RUN.POLY_SUBCHUNK_POINTS),
        "cuda_allow_tf32":
            bool(RUN.CUDA_ALLOW_TF32),
        "baseline_include_first_step_in_fit":
            bool(cfg.BASELINE_INCLUDE_FIRST_STEP_IN_FIT),
        "use_sample_weight":
            bool(cfg.USE_SAMPLE_WEIGHT),
    }
    _save_json(
        paths.config_snapshot,
        snapshot,
    )

    checkpoint = {
        "schema_name":
            "CPFE_STEP10_CLASSICAL_POLYNOMIAL2_OLS_V1",
        "created_time":
            _now_string(),

        "model_state_dict":
            {
                key: value.detach().cpu()
                for key, value
                in model.state_dict().items()
            },

        "phase":
            str(cfg.PHASE),
        "multiplier":
            int(cfg.MULTIPLIER),
        "model_type":
            "polynomial2",
        "degree":
            2,

        "input_dim":
            INPUT_DIM,
        "polynomial_feature_dim":
            POLY_DIM,
        "target_dim":
            TARGET_DIM,

        "fit_method":
            "streamed_weighted_OLS_on_degree2_polynomial_features",
        "regularization":
            "none",
        "fit_intercept":
            True,

        "fit_point_count":
            int(fit["point_count"]),
        "fit_weight_sum":
            float(fit["weight_sum"]),
        "gram_rank":
            int(fit["rank"]),
        "gram_condition_number":
            float(
                fit["condition_number"]
            ),

        "normalization_path":
            str(paths.normalization),
        "feature_map_path":
            str(paths.feature_map_json),

        "feature_order":
            list(POLY_FEATURE_NAMES),
        "second_order_pair_i":
            [
                int(x)
                for x in PAIR_I
            ],
        "second_order_pair_j":
            [
                int(x)
                for x in PAIR_J
            ],

        "train_metrics":
            train_metrics,
        "val_metrics":
            val_metrics,
        "lcs_line_pct":
            float(
                lcs.horizontal_line_pct
            ),
    }

    for path in (
        paths.best_checkpoint,
        paths.last_checkpoint,
    ):
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
            "CPFE_STEP10_CLASSICAL_POLYNOMIAL2_SUMMARY_V1",
        "created_time":
            _now_string(),

        "phase":
            str(cfg.PHASE),
        "multiplier":
            int(cfg.MULTIPLIER),

        "model_type":
            "polynomial2",
        "degree":
            2,

        "original_input_dim":
            INPUT_DIM,
        "polynomial_feature_dim":
            POLY_DIM,
        "target_dim":
            TARGET_DIM,

        "fit_method":
            "streamed_weighted_OLS_on_degree2_polynomial_features",
        "uses_optimizer":
            False,
        "uses_epochs":
            False,
        "uses_sequence_history":
            False,
        "regularization":
            "none",

        "fit_point_count":
            int(fit["point_count"]),
        "gram_rank":
            int(fit["rank"]),
        "gram_condition_number":
            float(
                fit["condition_number"]
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
            val_error < lcs_error,

        "best_checkpoint":
            str(
                paths.best_checkpoint
            ),
        "normalization":
            str(
                paths.normalization
            ),
        "feature_map":
            str(
                paths.feature_map_json
            ),
        "model_structure":
            str(
                paths.structure_json
            ),
    }
    _save_json(
        paths.summary_json,
        summary,
    )

    with paths.result_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "phase",
                "multiplier",
                "model",
                "degree",
                "input_dim",
                "poly_dim",
                "train_error_pct",
                "val_error_pct",
                "lcs_pct",
                "val_minus_lcs_pct_point",
                "below_lcs",
                "fit_points",
                "gram_rank",
                "gram_condition_number",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "phase":
                    cfg.PHASE,
                "multiplier":
                    cfg.MULTIPLIER,
                "model":
                    "polynomial2",
                "degree":
                    2,
                "input_dim":
                    INPUT_DIM,
                "poly_dim":
                    POLY_DIM,
                "train_error_pct":
                    train_metrics[
                        "weighted_global_relative_l2_pct"
                    ],
                "val_error_pct":
                    val_error,
                "lcs_pct":
                    lcs_error,
                "val_minus_lcs_pct_point":
                    val_error - lcs_error,
                "below_lcs":
                    val_error < lcs_error,
                "fit_points":
                    fit["point_count"],
                "gram_rank":
                    fit["rank"],
                "gram_condition_number":
                    fit["condition_number"],
            }
        )

    print("=" * 108)
    print("CORRECT SECOND-ORDER POLYNOMIAL REGRESSION COMPLETE")
    print("Phase             :", cfg.PHASE)
    print("Multiplier        :", cfg.MULTIPLIER)
    print("Original inputs   :", INPUT_DIM)
    print("Polynomial inputs :", POLY_DIM)
    print("Degree            : 2")
    print("Regularization    : NONE")
    print("Optimizer         : NONE")
    print("Epochs            : NONE")
    print("Fit points        :", f"{fit['point_count']:,}")
    print("Gram rank         :", fit["rank"])
    print(
        "Condition number  :",
        f"{fit['condition_number']:.6e}",
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
        "Checkpoint        :",
        paths.best_checkpoint,
    )
    print(
        "Feature map       :",
        paths.feature_map_json,
    )
    print(
        "Summary           :",
        paths.summary_json,
    )
    print(
        "Classical CSV     :",
        paths.result_csv,
    )
    print("=" * 108)


# =============================================================================
# 9. One phase
# =============================================================================

def run_phase(
    phase: str,
) -> None:
    # Use MODEL_TYPE="linear" ONLY to reuse the exact same source H5,
    # train/val selection, LCS reference and model-independent normalization
    # machinery. Polynomial output paths are constructed independently above.
    cfg = dataclasses.replace(
        BASE_CFG,
        PHASE=str(
            phase
        ).upper(),
        MODEL_TYPE="linear",
        MULTIPLIER=int(
            RUN.MULTIPLIER
        ),
        DEVICE="cpu",
        DATA_ACCESS_MODE="h5",
        NORMALIZATION_OVERWRITE=bool(
            RUN.NORMALIZATION_OVERWRITE
        ),
        ENABLE_LIVE_DASHBOARD=False,
        DASHBOARD_SHOW_WINDOW=False,
        RESUME_MODE="never",
    )

    cfg.validate()
    cfg.validate_required_files()

    info = inspect_phase_h5(
        cfg
    )
    lcs = load_lcs_reference(
        cfg
    )

    paths = _build_output_paths(
        cfg
    )
    _check_clean_output(
        paths
    )

    device = _resolve_device()

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(
            RUN.CUDA_ALLOW_TF32
        )

    print("=" * 108)
    print("Second-order Polynomial Regression")
    print("Phase             :", cfg.PHASE)
    print("Multiplier        :", cfg.MULTIPLIER)
    print("Sequence length   :", info.sequence_length)
    print("Train trajectories:", f"{info.train_count:,}")
    print("Val trajectories  :", f"{info.val_count:,}")
    print("Original inputs   :", INPUT_DIM)
    print("Second-order terms:", SECOND_ORDER_DIM)
    print("Polynomial inputs :", POLY_DIM)
    print("Target outputs    :", TARGET_DIM)
    print("Fit first step    :", cfg.BASELINE_INCLUDE_FIRST_STEP_IN_FIT)
    print("Sample weighting  :", cfg.USE_SAMPLE_WEIGHT)
    print("Regularization    : NONE")
    print("Optimizer         : NONE")
    print("Epochs            : NONE")
    print("Solve             : weighted OLS on degree-2 polynomial features")
    print("Accumulator device:", device)
    print("Subchunk points   :", f"{RUN.POLY_SUBCHUNK_POINTS:,}")
    print("Output root       :", paths.root)
    print("=" * 108)

    if RUN.CHECK_ONLY:
        print(
            "[CHECK ONLY] Paths, H5, LCS and Polynomial-2 settings are valid."
        )
        return

    stats = _prepare_normalization(
        cfg,
        paths,
    )

    fit = fit_weighted_poly2(
        cfg,
        stats,
        device,
    )

    train_metrics = evaluate_poly2(
        cfg,
        stats,
        fit,
        split="train",
        device=device,
    )
    val_metrics = evaluate_poly2(
        cfg,
        stats,
        fit,
        split="val",
        device=device,
    )

    model = Polynomial2Model().to(
        device
    )

    save_outputs(
        cfg=cfg,
        paths=paths,
        model=model,
        fit=fit,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        lcs=lcs,
        device=device,
    )


# =============================================================================
# 10. Main
# =============================================================================

def main() -> int:
    if int(
        RUN.POLY_SUBCHUNK_POINTS
    ) <= 0:
        raise ValueError(
            "POLY_SUBCHUNK_POINTS must be positive"
        )

    for phase in RUN.PHASES:
        run_phase(
            phase
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
