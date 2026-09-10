#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_9_train_ridge_bcc_hcp_FIXED.py

Correct classical weighted Ridge Regression baseline for Step-10.

This file DOES NOT call 10_7_train.py or 10_6_trainer.py.
Therefore Ridge is NOT trained with Adam/AdamW epochs.

It reuses the same Step-08 pointwise data and train-only normalization as the
correct Linear baseline, accumulates weighted sufficient statistics, solves one
closed-form Ridge system, evaluates Train/Val, and saves standard checkpoints.

Objective exactly matching 10_5_loss_and_metrics.py:
    weighted_MSE + RIDGE_ALPHA * sum(W^2)

Bias/intercept is NOT regularized.
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


@dataclass(frozen=True)
class LauncherConfig:
    PHASES: Tuple[str, ...] = ("BCC", "HCP")
    MULTIPLIER: int = 1

    # None -> read RIDGE_ALPHA from 10_1_config.py.
    RIDGE_ALPHA_OVERRIDE: float | None = None

    DEVICE: str = "cpu"

    # Prefer copying the already-validated Linear normalization for the same
    # phase/multiplier instead of recomputing it.
    REUSE_LINEAR_NORMALIZATION: bool = True

    ALLOW_OVERWRITE_EXISTING_RESULTS: bool = False
    NORMALIZATION_OVERWRITE: bool = False
    CHECK_ONLY: bool = False


RUN = LauncherConfig()

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


_cfg_mod = _import_local("step10_cfg_for_ridge_fixed", "10_1_config.py")
_data_mod = _import_local("step10_data_for_ridge_fixed", "10_2_h5_dataset.py")
_norm_mod = _import_local("step10_norm_for_ridge_fixed", "10_3_normalizer.py")
_model_mod = _import_local("step10_model_for_ridge_fixed", "10_4_models.py")

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


def _now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Progress:
    def __init__(self, label: str, total: int) -> None:
        self.label = str(label)
        self.total = max(1, int(total))
        self.current = 0
        self.started = time.perf_counter()

    def update(self, amount: int) -> None:
        self.current = min(self.total, self.current + int(amount))
        ratio = self.current / float(self.total)
        width = 32
        filled = min(width, int(round(width * ratio)))
        bar = "#" * filled + "-" * (width - filled)

        elapsed = time.perf_counter() - self.started
        rate = self.current / max(elapsed, 1.0e-12)
        remaining = (
            (self.total - self.current) / rate
            if rate > 0.0 else float("inf")
        )

        sys.stdout.write(
            f"\r{self.label:<16s} [{bar}] {100.0*ratio:6.2f}% "
            f"{self.current:,}/{self.total:,} points "
            f"| {rate:,.0f} point/s | ETA {remaining:,.1f}s"
        )
        sys.stdout.flush()

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
        x - np.asarray(stats.input_mean, dtype=np.float64)[None, :]
    ) / np.asarray(stats.input_std, dtype=np.float64)[None, :]


def _normalize_y(y, stats, cfg):
    y = np.asarray(y, dtype=np.float64)
    if not cfg.NORMALIZE_TARGETS:
        return y
    return (
        y - np.asarray(stats.target_mean, dtype=np.float64)[None, :]
    ) / np.asarray(stats.target_std, dtype=np.float64)[None, :]


def _denormalize_y(y, stats, cfg):
    y = np.asarray(y, dtype=np.float64)
    if not cfg.NORMALIZE_TARGETS:
        return y
    return (
        y * np.asarray(stats.target_std, dtype=np.float64)[None, :]
        + np.asarray(stats.target_mean, dtype=np.float64)[None, :]
    )


def _weights(raw, cfg):
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    w = raw if cfg.USE_SAMPLE_WEIGHT else np.ones_like(raw)

    if not np.isfinite(w).all():
        raise ValueError("Regression weights contain NaN/Inf")
    if np.any(w <= 0.0):
        raise ValueError("Regression weights must be positive")

    return w


def _check_clean_output(cfg):
    possible_conflicts = (
        Path(cfg.BEST_CHECKPOINT_PATH),
        Path(cfg.LAST_CHECKPOINT_PATH),
        Path(cfg.HISTORY_CSV_PATH),
        Path(cfg.TRAINING_SUMMARY_PATH),
    )
    existing = [p for p in possible_conflicts if p.exists()]

    if existing and not RUN.ALLOW_OVERWRITE_EXISTING_RESULTS:
        raise FileExistsError(
            "Existing Ridge results were found. They may belong to the old "
            "epoch-based Ridge implementation.\nRename/remove the old Ridge "
            "experiment folder first, or set "
            "ALLOW_OVERWRITE_EXISTING_RESULTS=True.\n"
            + "\n".join(f"  - {p}" for p in existing)
        )


def _prepare_normalization(cfg):
    ridge_path = Path(cfg.NORMALIZATION_PATH)

    if ridge_path.is_file() and not RUN.NORMALIZATION_OVERWRITE:
        try:
            stats = load_normalization_stats(ridge_path, cfg)
        except Exception as exc:
            print("[NORMALIZATION] Existing Ridge file incompatible:", exc)
        else:
            print("[NORMALIZATION] Reusing Ridge normalization:", ridge_path)
            return stats

    if RUN.REUSE_LINEAR_NORMALIZATION and not RUN.NORMALIZATION_OVERWRITE:
        linear_cfg = dataclasses.replace(
            cfg,
            MODEL_TYPE="linear",
            RESUME_MODE="never",
            ENABLE_LIVE_DASHBOARD=False,
            DASHBOARD_SHOW_WINDOW=False,
        )
        linear_path = Path(linear_cfg.NORMALIZATION_PATH)

        if linear_path.is_file():
            try:
                load_normalization_stats(linear_path, linear_cfg)
            except Exception as exc:
                print("[NORMALIZATION] Linear file found but incompatible:", exc)
            else:
                ridge_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(linear_path, ridge_path)

                linear_json = linear_path.with_suffix(".json")
                ridge_json = ridge_path.with_suffix(".json")
                if linear_json.is_file():
                    shutil.copy2(linear_json, ridge_json)

                stats = load_normalization_stats(ridge_path, cfg)

                print("[NORMALIZATION] Reused Linear normalization")
                print("        source:", linear_path)
                print("        Ridge :", ridge_path)
                return stats

    print("[NORMALIZATION] Recomputing train-only normalization.")
    return compile_train_only_normalization(
        cfg,
        force_recompute=bool(RUN.NORMALIZATION_OVERWRITE),
    )


def fit_weighted_ridge(cfg, stats):
    alpha = float(cfg.RIDGE_ALPHA)
    if alpha < 0.0:
        raise ValueError("RIDGE_ALPHA cannot be negative")

    fit_intercept = bool(cfg.RIDGE_FIT_INTERCEPT)
    include_first = bool(cfg.BASELINE_INCLUDE_FIRST_STEP_IN_FIT)

    d = INPUT_DIM
    c = TARGET_DIM

    xtwx = np.zeros((d, d), dtype=np.float64)
    xtwy = np.zeros((d, c), dtype=np.float64)

    if fit_intercept:
        xtw1 = np.zeros(d, dtype=np.float64)
        one_tw_y = np.zeros(c, dtype=np.float64)
        one_tw_one = 0.0

    point_count = 0
    weight_sum = 0.0

    total_points = _selected_point_count(
        cfg, "train", include_first
    )
    progress = Progress("Ridge accumulate", total_points)

    for chunk in iter_baseline_point_chunks(
        split_name="train",
        cfg=cfg,
        include_first_step=include_first,
        max_points=cfg.BASELINE_MAX_TRAIN_POINTS,
    ):
        x = _normalize_x(chunk["x"], stats, cfg)
        y = _normalize_y(chunk["y"], stats, cfg)
        w = _weights(chunk["weight"], cfg)

        xtwx += x.T @ (w[:, None] * x)
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

    # 10_5 defines:
    # total_loss = weighted_MSE + alpha * sum(W^2)
    # weighted_MSE denominator = sum(w) * TARGET_DIM.
    #
    # After normalizing Gram/RHS by sum(w), exact equivalent slope penalty is:
    # alpha * TARGET_DIM.
    effective_lambda = alpha * float(TARGET_DIM)

    if fit_intercept:
        gram = np.zeros((d + 1, d + 1), dtype=np.float64)
        rhs = np.zeros((d + 1, c), dtype=np.float64)

        gram[:d, :d] = xtwx / weight_sum
        gram[:d, d] = xtw1 / weight_sum
        gram[d, :d] = xtw1 / weight_sum
        gram[d, d] = one_tw_one / weight_sum

        rhs[:d, :] = xtwy / weight_sum
        rhs[d, :] = one_tw_y / weight_sum

        regularizer = np.zeros_like(gram)
        regularizer[:d, :d] = np.eye(d, dtype=np.float64)

        system = gram + effective_lambda * regularizer
    else:
        gram = xtwx / weight_sum
        rhs = xtwy / weight_sum
        system = gram + effective_lambda * np.eye(d, dtype=np.float64)

    print("[RIDGE SOLVE] Closed-form weighted Ridge")
    print("              RIDGE_ALPHA             =", alpha)
    print("              TARGET_DIM              =", TARGET_DIM)
    print("              Effective lambda        =", effective_lambda)
    print("              Intercept regularized   = NO")

    try:
        beta = np.linalg.solve(system, rhs)
        solve_method = "np.linalg.solve"
    except np.linalg.LinAlgError:
        beta, _, _, _ = np.linalg.lstsq(system, rhs, rcond=None)
        solve_method = "np.linalg.lstsq_fallback"

    if fit_intercept:
        coef = beta[:d, :].T
        intercept = beta[d, :]
    else:
        coef = beta.T
        intercept = np.zeros(c, dtype=np.float64)

    singular_values = np.linalg.svd(system, compute_uv=False)
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )

    return {
        "coef": coef,
        "intercept": intercept,
        "alpha": alpha,
        "effective_lambda": effective_lambda,
        "point_count": int(point_count),
        "weight_sum": float(weight_sum),
        "rank": int(np.linalg.matrix_rank(system)),
        "condition_number": condition_number,
        "solve_method": solve_method,
    }


def evaluate_ridge(cfg, stats, fit, split: str):
    total_points = _selected_point_count(
        cfg, split, include_first=False
    )
    progress = Progress(f"Eval {split}", total_points)

    weighted_sse = 0.0
    weighted_target_sq = 0.0
    weighted_sse_comp = np.zeros(TARGET_DIM, dtype=np.float64)
    weighted_target_comp = np.zeros(TARGET_DIM, dtype=np.float64)

    normalized_sse = 0.0
    normalized_den = 0.0
    point_count = 0

    max_points = (
        cfg.BASELINE_MAX_TRAIN_POINTS
        if split == "train"
        else cfg.BASELINE_MAX_VAL_POINTS
    )

    coef = np.asarray(fit["coef"], dtype=np.float64)
    intercept = np.asarray(fit["intercept"], dtype=np.float64)

    for chunk in iter_baseline_point_chunks(
        split_name=split,
        cfg=cfg,
        include_first_step=False,
        max_points=max_points,
    ):
        x_norm = _normalize_x(chunk["x"], stats, cfg)
        y_phys = np.asarray(chunk["y"], dtype=np.float64)
        y_norm = _normalize_y(y_phys, stats, cfg)
        w = _weights(chunk["weight"], cfg)

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
        normalized_den += (
            float(np.sum(w, dtype=np.float64)) * TARGET_DIM
        )

        n = int(y_phys.shape[0])
        point_count += n
        progress.update(n)

    if point_count <= 0:
        raise RuntimeError(f"No {split} points evaluated")

    relative_l2 = (
        100.0 * math.sqrt(weighted_sse / weighted_target_sq)
        if weighted_target_sq > 0.0
        else float("nan")
    )

    component = np.full(TARGET_DIM, np.nan, dtype=np.float64)
    valid = weighted_target_comp > 0.0
    component[valid] = (
        100.0 * np.sqrt(
            weighted_sse_comp[valid] / weighted_target_comp[valid]
        )
    )

    return {
        "split": split,
        "point_count": int(point_count),
        "normalized_weighted_mse":
            float(normalized_sse / normalized_den),
        "weighted_global_relative_l2_pct": float(relative_l2),
        "weighted_component_relative_l2_pct":
            [float(x) for x in component],
        "component_names": list(TARGET_ORDER),
        "first_step_excluded": True,
    }


def save_outputs(cfg, model, fit, train_metrics, val_metrics, lcs):
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
        "schema_name": "CPFE_STEP10_CLASSICAL_RIDGE_V1",
        "created_time": _now_string(),
        "model_state_dict": model.state_dict(),
        "phase": str(cfg.PHASE),
        "multiplier": int(cfg.MULTIPLIER),
        "model_type": "ridge",
        "architecture_tag": str(cfg.MODEL_ARCHITECTURE_TAG),
        "input_dim": INPUT_DIM,
        "target_dim": TARGET_DIM,
        "fit_method": "streamed_weighted_closed_form_ridge",
        "fit_intercept": bool(cfg.RIDGE_FIT_INTERCEPT),
        "ridge_alpha": float(fit["alpha"]),
        "effective_lambda": float(fit["effective_lambda"]),
        "bias_regularized": False,
        "fit_point_count": int(fit["point_count"]),
        "fit_weight_sum": float(fit["weight_sum"]),
        "system_rank": int(fit["rank"]),
        "system_condition_number":
            float(fit["condition_number"]),
        "solve_method": str(fit["solve_method"]),
        "normalization_path": str(Path(cfg.NORMALIZATION_PATH)),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "lcs_line_pct": float(lcs.horizontal_line_pct),
        "config_snapshot": cfg.snapshot_dict(),
    }

    for raw in (
        cfg.BEST_CHECKPOINT_PATH,
        cfg.LAST_CHECKPOINT_PATH,
    ):
        path = Path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temp)
        os.replace(temp, path)

    val_error = float(
        val_metrics["weighted_global_relative_l2_pct"]
    )
    lcs_error = float(lcs.horizontal_line_pct)

    summary = {
        "schema_name": "CPFE_STEP10_CLASSICAL_RIDGE_SUMMARY_V1",
        "created_time": _now_string(),
        "phase": str(cfg.PHASE),
        "multiplier": int(cfg.MULTIPLIER),
        "model_type": "ridge",
        "fit_method": "streamed_weighted_closed_form_ridge",
        "uses_optimizer": False,
        "uses_epochs": False,
        "uses_sequence_history": False,
        "ridge_alpha": float(fit["alpha"]),
        "effective_lambda": float(fit["effective_lambda"]),
        "fit_intercept": bool(cfg.RIDGE_FIT_INTERCEPT),
        "bias_regularized": False,
        "fit_point_count": int(fit["point_count"]),
        "system_rank": int(fit["rank"]),
        "system_condition_number":
            float(fit["condition_number"]),
        "solve_method": str(fit["solve_method"]),
        "train": train_metrics,
        "val": val_metrics,
        "lcs_line_pct": lcs_error,
        "val_minus_lcs_pct_point": val_error - lcs_error,
        "below_lcs": val_error < lcs_error,
        "best_checkpoint": str(cfg.BEST_CHECKPOINT_PATH),
        "normalization": str(cfg.NORMALIZATION_PATH),
        "model_structure": str(structure_path),
    }

    Path(cfg.TRAINING_SUMMARY_PATH).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = Path(cfg.EXPERIMENT_ROOT) / "classical_fit_result.csv"
    with csv_path.open(
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
                "ridge_alpha",
                "effective_lambda",
                "train_error_pct",
                "val_error_pct",
                "lcs_pct",
                "val_minus_lcs_pct_point",
                "below_lcs",
                "fit_points",
                "system_rank",
                "system_condition_number",
                "solve_method",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "phase": cfg.PHASE,
                "multiplier": cfg.MULTIPLIER,
                "model": "ridge",
                "ridge_alpha": fit["alpha"],
                "effective_lambda": fit["effective_lambda"],
                "train_error_pct":
                    train_metrics["weighted_global_relative_l2_pct"],
                "val_error_pct": val_error,
                "lcs_pct": lcs_error,
                "val_minus_lcs_pct_point":
                    val_error - lcs_error,
                "below_lcs": val_error < lcs_error,
                "fit_points": fit["point_count"],
                "system_rank": fit["rank"],
                "system_condition_number":
                    fit["condition_number"],
                "solve_method": fit["solve_method"],
            }
        )

    if RUN.ALLOW_OVERWRITE_EXISTING_RESULTS:
        old_history = Path(cfg.HISTORY_CSV_PATH)
        if old_history.exists():
            old_history.unlink()

    print("=" * 108)
    print("CORRECT CLASSICAL RIDGE COMPLETE")
    print("Phase             :", cfg.PHASE)
    print("Multiplier        :", cfg.MULTIPLIER)
    print("RIDGE_ALPHA       :", fit["alpha"])
    print("Effective lambda  :", fit["effective_lambda"])
    print("Fit points        :", f"{fit['point_count']:,}")
    print("System rank       :", fit["rank"])
    print("Condition number  :", f"{fit['condition_number']:.6e}")
    print("Solve method      :", fit["solve_method"])
    print("Train error       : %.6f %%" %
          train_metrics["weighted_global_relative_l2_pct"])
    print("Validation error  : %.6f %%" % val_error)
    print("LCS line          : %.6f %%" % lcs_error)
    print("Val - LCS         : %.6f pct-point" %
          (val_error - lcs_error))
    print("Below LCS         :", "YES" if val_error < lcs_error else "NO")
    print("Best checkpoint   :", cfg.BEST_CHECKPOINT_PATH)
    print("Summary           :", cfg.TRAINING_SUMMARY_PATH)
    print("Classical CSV     :", csv_path)
    print("Epoch history     : NONE (correct for closed-form Ridge)")
    print("=" * 108)


def run_phase(phase: str):
    overrides = {
        "PHASE": str(phase).upper(),
        "MODEL_TYPE": "ridge",
        "MULTIPLIER": int(RUN.MULTIPLIER),
        "DEVICE": str(RUN.DEVICE),
        "DATA_ACCESS_MODE": "h5",
        "NORMALIZATION_OVERWRITE":
            bool(RUN.NORMALIZATION_OVERWRITE),
        "ENABLE_LIVE_DASHBOARD": False,
        "DASHBOARD_SHOW_WINDOW": False,
        "RESUME_MODE": "never",
    }

    if RUN.RIDGE_ALPHA_OVERRIDE is not None:
        overrides["RIDGE_ALPHA"] = float(
            RUN.RIDGE_ALPHA_OVERRIDE
        )

    cfg = dataclasses.replace(BASE_CFG, **overrides)

    cfg.validate()
    cfg.validate_required_files()

    info = inspect_phase_h5(cfg)
    lcs = load_lcs_reference(cfg)

    print("=" * 108)
    print("Correct classical Ridge Regression")
    print("Phase             :", cfg.PHASE)
    print("Multiplier        :", cfg.MULTIPLIER)
    print("Sequence length   :", info.sequence_length)
    print("Train trajectories:", f"{info.train_count:,}")
    print("Val trajectories  :", f"{info.val_count:,}")
    print("RIDGE_ALPHA       :", cfg.RIDGE_ALPHA)
    print("Fit intercept     :", cfg.RIDGE_FIT_INTERCEPT)
    print("Fit first step    :", cfg.BASELINE_INCLUDE_FIRST_STEP_IN_FIT)
    print("Optimizer         : NONE")
    print("Epochs            : NONE")
    print("Solve             : weighted closed-form Ridge")
    print("Output root       :", cfg.EXPERIMENT_ROOT)
    print("=" * 108)

    _check_clean_output(cfg)

    if RUN.CHECK_ONLY:
        print("[CHECK ONLY] Paths, H5, LCS and Ridge settings are valid.")
        return

    stats = _prepare_normalization(cfg)
    fit = fit_weighted_ridge(cfg, stats)

    train_metrics = evaluate_ridge(
        cfg, stats, fit, split="train"
    )
    val_metrics = evaluate_ridge(
        cfg, stats, fit, split="val"
    )

    model = build_model(cfg)

    save_outputs(
        cfg=cfg,
        model=model,
        fit=fit,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        lcs=lcs,
    )


def main() -> int:
    if (
        RUN.RIDGE_ALPHA_OVERRIDE is not None
        and RUN.RIDGE_ALPHA_OVERRIDE < 0.0
    ):
        raise ValueError(
            "RIDGE_ALPHA_OVERRIDE cannot be negative"
        )

    for phase in RUN.PHASES:
        run_phase(phase)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
