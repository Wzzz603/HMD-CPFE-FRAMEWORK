#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_9_train_linear_bcc_hcp_FIXED.py

Correct classical Linear Regression baseline for Step-10.

IMPORTANT
---------
This file DOES NOT call 10_7_train.py or 10_6_trainer.py.
Therefore Linear Regression is NOT trained with Adam/AdamW epochs.

For each phase:
    Step-08 H5
      -> iter_baseline_point_chunks()
      -> train-only normalization
      -> streamed weighted sufficient statistics
      -> one weighted ordinary least-squares solve
      -> final train/validation evaluation
      -> best_checkpoint.pt / last_checkpoint.pt

Model form:
    y_hat = W x + b

Fit objective in normalized target space:
    argmin sum_i w_i || y_i - (W x_i + b) ||^2

The first time point IS included in fitting when
BASELINE_INCLUDE_FIRST_STEP_IN_FIT=True, but final LCS-comparable Train/Val
metrics exclude the first time point, matching Step-09.

Put this script in the same folder as:
    10_1_config.py
    10_2_h5_dataset.py
    10_3_normalizer.py
    10_4_models.py
    10_5_loss_and_metrics.py

Before the first correct run, rename/remove the old experiment folders produced
by the erroneous epoch-based Linear run, otherwise this script stops to avoid
mixing incompatible results.
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import torch


# =============================================================================
# 0. USER CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class LauncherConfig:
    PHASES: Tuple[str, ...] = ("BCC", "HCP")
    MULTIPLIER: int = 1

    # Classical OLS is solved on CPU in float64.
    DEVICE: str = "cpu"

    # Safety: do not overwrite the old incorrect epoch-trained Linear results.
    # Rename/remove the old experiment folder first.
    ALLOW_OVERWRITE_EXISTING_RESULTS: bool = False

    # Usually False. If a compatible normalization file is present in a clean
    # experiment folder, it is reused.
    NORMALIZATION_OVERWRITE: bool = False

    # If True, only validate and print what would be done.
    CHECK_ONLY: bool = False


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


_cfg_mod = _import_local("step10_cfg_for_linear_ols", "10_1_config.py")
_data_mod = _import_local("step10_data_for_linear_ols", "10_2_h5_dataset.py")
_norm_mod = _import_local("step10_norm_for_linear_ols", "10_3_normalizer.py")
_model_mod = _import_local("step10_model_for_linear_ols", "10_4_models.py")

BASE_CFG = _cfg_mod.CFG
INPUT_DIM = int(_cfg_mod.INPUT_DIM)
TARGET_DIM = int(_cfg_mod.TARGET_DIM)
TARGET_ORDER = tuple(_cfg_mod.TARGET_ORDER)

inspect_phase_h5 = _data_mod.inspect_phase_h5
load_lcs_reference = _data_mod.load_lcs_reference
iter_baseline_point_chunks = _data_mod.iter_baseline_point_chunks

compile_train_only_normalization = _norm_mod.compile_train_only_normalization
build_model = _model_mod.build_model
save_model_structure = _model_mod.save_model_structure
get_model_info = _model_mod.get_model_info


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
        self.last_width = 0

    def update(self, amount: int) -> None:
        self.current += int(amount)
        self.current = min(self.current, self.total)

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

        text = (
            f"\r{self.label:<16s} "
            f"[{bar}] {100.0*ratio:6.2f}% "
            f"{self.current:,}/{self.total:,} points "
            f"| {rate:,.0f} point/s "
            f"| ETA {remaining:,.1f}s"
        )
        sys.stdout.write(text)
        sys.stdout.flush()
        self.last_width = len(text)

        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _selected_point_count(cfg, split: str, include_first: bool) -> int:
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

    ntime = int(info.sequence_length) if include_first else int(info.sequence_length) - 1
    total = ntraj * ntime

    if max_points is not None:
        total = min(total, int(max_points))

    return int(total)


def _normalize_x(x: np.ndarray, stats, cfg) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not bool(cfg.NORMALIZE_INPUTS):
        return x
    return (
        x
        - np.asarray(stats.input_mean, dtype=np.float64)[None, :]
    ) / np.asarray(stats.input_std, dtype=np.float64)[None, :]


def _normalize_y(y: np.ndarray, stats, cfg) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if not bool(cfg.NORMALIZE_TARGETS):
        return y
    return (
        y
        - np.asarray(stats.target_mean, dtype=np.float64)[None, :]
    ) / np.asarray(stats.target_std, dtype=np.float64)[None, :]


def _denormalize_y(y: np.ndarray, stats, cfg) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if not bool(cfg.NORMALIZE_TARGETS):
        return y
    return (
        y
        * np.asarray(stats.target_std, dtype=np.float64)[None, :]
        + np.asarray(stats.target_mean, dtype=np.float64)[None, :]
    )


def _effective_weights(raw: np.ndarray, cfg) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    if bool(cfg.USE_SAMPLE_WEIGHT):
        w = raw
    else:
        w = np.ones_like(raw)

    if not np.isfinite(w).all() or np.any(w <= 0.0):
        raise ValueError("Invalid regression weights")
    return w


def _check_clean_output(cfg, allow_overwrite: bool) -> None:
    root = Path(cfg.EXPERIMENT_ROOT)

    conflicting = [
        Path(cfg.BEST_CHECKPOINT_PATH),
        Path(cfg.LAST_CHECKPOINT_PATH),
        Path(cfg.HISTORY_CSV_PATH),  # old epoch history is especially unsafe
        Path(cfg.TRAINING_SUMMARY_PATH),
    ]
    existing = [p for p in conflicting if p.exists()]

    if existing and not allow_overwrite:
        text = "\n".join(f"  - {p}" for p in existing)
        raise FileExistsError(
            "Existing Linear training results were found. They may belong to "
            "the incorrect epoch-based run.\n"
            "Rename/remove the old experiment folder before the correct OLS "
            "run, or explicitly set ALLOW_OVERWRITE_EXISTING_RESULTS=True.\n"
            f"Experiment root: {root}\nExisting:\n{text}"
        )


# =============================================================================
# 3. Weighted OLS fit
# =============================================================================

def fit_weighted_ols(cfg, stats) -> Dict[str, object]:
    fit_intercept = bool(cfg.LINEAR_FIT_INTERCEPT)
    include_first = bool(cfg.BASELINE_INCLUDE_FIRST_STEP_IN_FIT)

    d = INPUT_DIM
    c = TARGET_DIM

    # Sufficient statistics in float64.
    xtwx = np.zeros((d, d), dtype=np.float64)
    xtwy = np.zeros((d, c), dtype=np.float64)

    if fit_intercept:
        xtw1 = np.zeros(d, dtype=np.float64)
        one_tw_y = np.zeros(c, dtype=np.float64)
        one_tw_one = 0.0

    point_count = 0
    weight_sum = 0.0

    total_points = _selected_point_count(
        cfg, split="train", include_first=include_first
    )
    progress = Progress("OLS accumulate", total_points)

    for chunk in iter_baseline_point_chunks(
        split_name="train",
        cfg=cfg,
        include_first_step=include_first,
        max_points=cfg.BASELINE_MAX_TRAIN_POINTS,
    ):
        x = _normalize_x(chunk["x"], stats, cfg)
        y = _normalize_y(chunk["y"], stats, cfg)
        w = _effective_weights(chunk["weight"], cfg)

        if x.shape != (x.shape[0], d):
            raise RuntimeError(f"Unexpected x shape {x.shape}")
        if y.shape != (x.shape[0], c):
            raise RuntimeError(f"Unexpected y shape {y.shape}")

        # Avoid constructing diag(w):
        # X^T W X = X^T (w[:,None] * X)
        wx = w[:, None] * x
        xtwx += x.T @ wx
        xtwy += x.T @ (w[:, None] * y)

        if fit_intercept:
            xtw1 += x.T @ w
            one_tw_y += w @ y
            one_tw_one += float(np.sum(w, dtype=np.float64))

        n = int(x.shape[0])
        point_count += n
        weight_sum += float(np.sum(w, dtype=np.float64))
        progress.update(n)

    if point_count <= 0 or weight_sum <= 0.0:
        raise RuntimeError("No training points accumulated")

    if fit_intercept:
        gram = np.zeros((d + 1, d + 1), dtype=np.float64)
        rhs = np.zeros((d + 1, c), dtype=np.float64)

        gram[:d, :d] = xtwx
        gram[:d, d] = xtw1
        gram[d, :d] = xtw1
        gram[d, d] = one_tw_one

        rhs[:d, :] = xtwy
        rhs[d, :] = one_tw_y
    else:
        gram = xtwx
        rhs = xtwy

    # Common scaling keeps matrix magnitudes moderate and changes neither
    # minimizer nor rank.
    gram /= weight_sum
    rhs /= weight_sum

    print("[OLS SOLVE] Solving weighted normal equations with np.linalg.lstsq ...")
    beta, residuals, rank, singular_values = np.linalg.lstsq(
        gram,
        rhs,
        rcond=None,
    )

    if fit_intercept:
        coef = beta[:d, :].T       # [6,22]
        intercept = beta[d, :]     # [6]
    else:
        coef = beta.T
        intercept = np.zeros(c, dtype=np.float64)

    if coef.shape != (c, d):
        raise RuntimeError(f"Invalid coefficient shape {coef.shape}")

    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )

    return {
        "coef": coef,
        "intercept": intercept,
        "point_count": int(point_count),
        "weight_sum": float(weight_sum),
        "rank": int(rank),
        "condition_number": condition_number,
        "singular_values": singular_values,
    }


# =============================================================================
# 4. Evaluation
# =============================================================================

def evaluate_linear(cfg, stats, coef, intercept, split: str) -> Dict[str, object]:
    # Strict LCS-comparable reporting excludes time index 0.
    include_first = False

    total_points = _selected_point_count(
        cfg, split=split, include_first=include_first
    )
    progress = Progress(f"Eval {split}", total_points)

    weighted_sse = 0.0
    weighted_target_sq = 0.0
    weighted_sse_comp = np.zeros(TARGET_DIM, dtype=np.float64)
    weighted_target_comp = np.zeros(TARGET_DIM, dtype=np.float64)

    normalized_sse = 0.0
    normalized_den = 0.0

    points = 0

    max_points = (
        cfg.BASELINE_MAX_TRAIN_POINTS
        if split == "train"
        else cfg.BASELINE_MAX_VAL_POINTS
    )

    for chunk in iter_baseline_point_chunks(
        split_name=split,
        cfg=cfg,
        include_first_step=False,
        max_points=max_points,
    ):
        x_norm = _normalize_x(chunk["x"], stats, cfg)
        y_phys = np.asarray(chunk["y"], dtype=np.float64)
        y_norm = _normalize_y(y_phys, stats, cfg)
        w = _effective_weights(chunk["weight"], cfg)

        pred_norm = x_norm @ coef.T + intercept[None, :]
        pred_phys = _denormalize_y(pred_norm, stats, cfg)

        err = pred_phys - y_phys

        weighted_sse += float(
            np.sum(w[:, None] * err * err, dtype=np.float64)
        )
        weighted_target_sq += float(
            np.sum(w[:, None] * y_phys * y_phys, dtype=np.float64)
        )
        weighted_sse_comp += np.sum(
            w[:, None] * err * err,
            axis=0,
            dtype=np.float64,
        )
        weighted_target_comp += np.sum(
            w[:, None] * y_phys * y_phys,
            axis=0,
            dtype=np.float64,
        )

        norm_err = pred_norm - y_norm
        normalized_sse += float(
            np.sum(w[:, None] * norm_err * norm_err, dtype=np.float64)
        )
        normalized_den += float(np.sum(w, dtype=np.float64)) * TARGET_DIM

        n = int(y_phys.shape[0])
        points += n
        progress.update(n)

    if points <= 0:
        raise RuntimeError(f"No {split} points evaluated")

    relative_l2 = (
        100.0 * math.sqrt(weighted_sse / weighted_target_sq)
        if weighted_target_sq > 0.0
        else float("nan")
    )

    component = np.full(TARGET_DIM, np.nan, dtype=np.float64)
    valid = weighted_target_comp > 0.0
    component[valid] = (
        100.0
        * np.sqrt(weighted_sse_comp[valid] / weighted_target_comp[valid])
    )

    return {
        "split": split,
        "point_count": int(points),
        "normalized_weighted_mse": float(normalized_sse / normalized_den),
        "weighted_global_relative_l2_pct": float(relative_l2),
        "weighted_component_relative_l2_pct": [
            float(x) for x in component
        ],
        "component_names": list(TARGET_ORDER),
        "first_step_excluded": True,
    }


# =============================================================================
# 5. Save artifacts
# =============================================================================

def save_outputs(cfg, model, stats, fit, train_metrics, val_metrics, lcs) -> None:
    cfg.create_output_directories()
    cfg.save_snapshot()

    coef = np.asarray(fit["coef"], dtype=np.float64)
    intercept = np.asarray(fit["intercept"], dtype=np.float64)

    with torch.no_grad():
        model.regression_weight.copy_(
            torch.as_tensor(
                coef,
                dtype=model.regression_weight.dtype,
                device=model.regression_weight.device,
            )
        )
        if model.regression_bias is not None:
            model.regression_bias.copy_(
                torch.as_tensor(
                    intercept,
                    dtype=model.regression_bias.dtype,
                    device=model.regression_bias.device,
                )
            )

    structure_path = save_model_structure(model, cfg)

    checkpoint = {
        "schema_name": "CPFE_STEP10_CLASSICAL_LINEAR_OLS_V1",
        "created_time": _now_string(),
        "model_state_dict": model.state_dict(),
        "phase": str(cfg.PHASE),
        "multiplier": int(cfg.MULTIPLIER),
        "model_type": "linear",
        "architecture_tag": str(cfg.MODEL_ARCHITECTURE_TAG),
        "input_dim": INPUT_DIM,
        "target_dim": TARGET_DIM,
        "fit_method": "streamed_weighted_ordinary_least_squares",
        "fit_intercept": bool(cfg.LINEAR_FIT_INTERCEPT),
        "fit_point_count": int(fit["point_count"]),
        "fit_weight_sum": float(fit["weight_sum"]),
        "gram_rank": int(fit["rank"]),
        "gram_condition_number": float(fit["condition_number"]),
        "normalization_path": str(Path(cfg.NORMALIZATION_PATH)),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "lcs_line_pct": float(lcs.horizontal_line_pct),
        "config_snapshot": cfg.snapshot_dict(),
    }

    for raw in (cfg.BEST_CHECKPOINT_PATH, cfg.LAST_CHECKPOINT_PATH):
        path = Path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temp)
        os.replace(temp, path)

    summary = {
        "schema_name": "CPFE_STEP10_CLASSICAL_LINEAR_OLS_SUMMARY_V1",
        "created_time": _now_string(),
        "phase": str(cfg.PHASE),
        "multiplier": int(cfg.MULTIPLIER),
        "model_type": "linear",
        "fit_method": "streamed_weighted_ordinary_least_squares",
        "uses_optimizer": False,
        "uses_epochs": False,
        "uses_sequence_history": False,
        "fit_intercept": bool(cfg.LINEAR_FIT_INTERCEPT),
        "fit_point_count": int(fit["point_count"]),
        "gram_rank": int(fit["rank"]),
        "gram_condition_number": float(fit["condition_number"]),
        "train": train_metrics,
        "val": val_metrics,
        "lcs_line_pct": float(lcs.horizontal_line_pct),
        "val_minus_lcs_pct_point": (
            float(val_metrics["weighted_global_relative_l2_pct"])
            - float(lcs.horizontal_line_pct)
        ),
        "below_lcs": (
            float(val_metrics["weighted_global_relative_l2_pct"])
            < float(lcs.horizontal_line_pct)
        ),
        "best_checkpoint": str(cfg.BEST_CHECKPOINT_PATH),
        "normalization": str(cfg.NORMALIZATION_PATH),
        "model_structure": str(structure_path),
    }

    Path(cfg.TRAINING_SUMMARY_PATH).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # No epoch history is created. Save a one-row classical fit result instead.
    csv_path = Path(cfg.EXPERIMENT_ROOT) / "classical_fit_result.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "phase",
                "multiplier",
                "model",
                "fit_method",
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
                "phase": cfg.PHASE,
                "multiplier": cfg.MULTIPLIER,
                "model": "linear",
                "fit_method": "weighted_OLS",
                "train_error_pct":
                    train_metrics["weighted_global_relative_l2_pct"],
                "val_error_pct":
                    val_metrics["weighted_global_relative_l2_pct"],
                "lcs_pct": lcs.horizontal_line_pct,
                "val_minus_lcs_pct_point":
                    summary["val_minus_lcs_pct_point"],
                "below_lcs": summary["below_lcs"],
                "fit_points": fit["point_count"],
                "gram_rank": fit["rank"],
                "gram_condition_number": fit["condition_number"],
            }
        )

    # Remove stale epoch history only when explicit overwrite was requested.
    # In normal safe use the script refuses to run if it exists.
    if RUN.ALLOW_OVERWRITE_EXISTING_RESULTS:
        history_path = Path(cfg.HISTORY_CSV_PATH)
        if history_path.exists():
            history_path.unlink()

    print("=" * 108)
    print("CORRECT CLASSICAL LINEAR OLS COMPLETE")
    print("Phase             :", cfg.PHASE)
    print("Multiplier        :", cfg.MULTIPLIER)
    print("Fit points        :", f"{fit['point_count']:,}")
    print("Gram rank         :", fit["rank"])
    print("Condition number  :", f"{fit['condition_number']:.6e}")
    print("Train error       : %.6f %%" %
          train_metrics["weighted_global_relative_l2_pct"])
    print("Validation error  : %.6f %%" %
          val_metrics["weighted_global_relative_l2_pct"])
    print("LCS line          : %.6f %%" % lcs.horizontal_line_pct)
    print("Val - LCS         : %.6f pct-point" %
          summary["val_minus_lcs_pct_point"])
    print("Below LCS         :", "YES" if summary["below_lcs"] else "NO")
    print("Best checkpoint   :", cfg.BEST_CHECKPOINT_PATH)
    print("Summary           :", cfg.TRAINING_SUMMARY_PATH)
    print("Classical CSV     :", csv_path)
    print("Epoch history     : NONE (correct for closed-form Linear)")
    print("=" * 108)


# =============================================================================
# 6. One phase
# =============================================================================

def run_phase(phase: str) -> None:
    cfg = dataclasses.replace(
        BASE_CFG,
        PHASE=str(phase).upper(),
        MODEL_TYPE="linear",
        MULTIPLIER=int(RUN.MULTIPLIER),
        DEVICE=str(RUN.DEVICE),
        DATA_ACCESS_MODE="h5",
        NORMALIZATION_OVERWRITE=bool(RUN.NORMALIZATION_OVERWRITE),
        ENABLE_LIVE_DASHBOARD=False,
        DASHBOARD_SHOW_WINDOW=False,
        RESUME_MODE="never",
    )
    cfg.validate()
    cfg.validate_required_files()

    info = inspect_phase_h5(cfg)
    lcs = load_lcs_reference(cfg)

    print("=" * 108)
    print("Correct classical Linear Regression")
    print("Phase             :", cfg.PHASE)
    print("Multiplier        :", cfg.MULTIPLIER)
    print("Sequence length   :", info.sequence_length)
    print("Train trajectories:", f"{info.train_count:,}")
    print("Val trajectories  :", f"{info.val_count:,}")
    print("Fit intercept     :", cfg.LINEAR_FIT_INTERCEPT)
    print("Fit first step    :", cfg.BASELINE_INCLUDE_FIRST_STEP_IN_FIT)
    print("Optimizer         : NONE")
    print("Epochs            : NONE")
    print("Solve             : weighted OLS / np.linalg.lstsq")
    print("Output root       :", cfg.EXPERIMENT_ROOT)
    print("=" * 108)

    _check_clean_output(
        cfg,
        allow_overwrite=RUN.ALLOW_OVERWRITE_EXISTING_RESULTS,
    )

    if RUN.CHECK_ONLY:
        print("[CHECK ONLY] Paths and H5/LCS configuration are valid.")
        return

    stats = compile_train_only_normalization(
        cfg,
        force_recompute=bool(RUN.NORMALIZATION_OVERWRITE),
    )

    fit = fit_weighted_ols(cfg, stats)

    train_metrics = evaluate_linear(
        cfg,
        stats,
        np.asarray(fit["coef"], dtype=np.float64),
        np.asarray(fit["intercept"], dtype=np.float64),
        split="train",
    )
    val_metrics = evaluate_linear(
        cfg,
        stats,
        np.asarray(fit["coef"], dtype=np.float64),
        np.asarray(fit["intercept"], dtype=np.float64),
        split="val",
    )

    model = build_model(cfg)
    save_outputs(
        cfg=cfg,
        model=model,
        stats=stats,
        fit=fit,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        lcs=lcs,
    )


# =============================================================================
# 7. Main
# =============================================================================

def main() -> int:
    for phase in RUN.PHASES:
        run_phase(phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
