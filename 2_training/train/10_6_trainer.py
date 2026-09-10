#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_6_trainer.py

Unified Step-10 trainer with complete resume restoration.

Key resume guarantee
--------------------
Every completed epoch is saved atomically to last_checkpoint.pt together with:

- model state;
- optimizer state;
- LR-scheduler state;
- AMP scaler state;
- complete epoch history from epoch 1;
- LCS first-crossing and best-error state;
- early-stopping state;
- cumulative training time;
- Python/NumPy/PyTorch RNG states;
- DataLoader shuffle-generator state.

When training resumes, the live table, CSV and error chart are rebuilt from the
complete saved history before the next epoch starts. Therefore, a resumed run
continues the original overall curves instead of starting a new chart.

If interruption occurs inside an epoch, automatic continuation returns to the
last fully completed epoch. The unfinished epoch is deliberately not merged
with the history because its train/validation statistics are incomplete.

Models
------
- LSTM: two stacked recurrent layers plus one Linear(256,6) output head.
- Linear/Ridge/Lasso: same pointwise affine map at every time step.
- No physics-informed loss.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


# =============================================================================
# 0. Dynamic imports
# =============================================================================

_THIS_DIR = Path(__file__).resolve().parent


def _import_local_module(module_name: str, filename: str):
    path = _THIS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            "%s must be in the same directory: %s"
            % (filename, path)
        )

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not create import specification for %s" % path
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_config_module = _import_local_module(
    "step10_config_for_trainer",
    "10_1_config.py",
)
_dataset_module = _import_local_module(
    "step10_dataset_for_trainer",
    "10_2_h5_dataset.py",
)
_normalizer_module = _import_local_module(
    "step10_normalizer_for_trainer",
    "10_3_normalizer.py",
)
_models_module = _import_local_module(
    "step10_models_for_trainer",
    "10_4_models.py",
)
_loss_module = _import_local_module(
    "step10_loss_for_trainer",
    "10_5_loss_and_metrics.py",
)

CFG = _config_module.CFG
TARGET_ORDER = tuple(_config_module.TARGET_ORDER)

build_lstm_data_bundle = _dataset_module.build_lstm_data_bundle
LSTMDataBundle = _dataset_module.LSTMDataBundle

compile_train_only_normalization = (
    _normalizer_module.compile_train_only_normalization
)
TorchNormalizer = _normalizer_module.TorchNormalizer

build_model = _models_module.build_model
get_model_info = _models_module.get_model_info
save_model_structure = _models_module.save_model_structure

compute_training_loss = _loss_module.compute_training_loss
regression_regularization_penalty = (
    _loss_module.regression_regularization_penalty
)
StressMetricAccumulator = _loss_module.StressMetricAccumulator
update_physical_metric_from_normalized = (
    _loss_module.update_physical_metric_from_normalized
)
LCSCrossingTracker = _loss_module.LCSCrossingTracker
build_epoch_record = _loss_module.build_epoch_record
save_metric_definition = _loss_module.save_metric_definition


# =============================================================================
# 1. Small helpers
# =============================================================================

HISTORY_FIELDS = (
    "epoch",
    "elapsed_seconds",
    "epoch_seconds",
    "learning_rate",

    "train_total_loss",
    "train_data_loss",
    "train_regularization_loss",
    "train_metric_pct",

    "val_total_loss",
    "val_data_loss",
    "val_regularization_loss",
    "val_metric_pct",

    "lcs_line_pct",
    "val_minus_lcs_pct_point",
    "is_below_lcs",
    "just_crossed_lcs",
    "first_below_lcs_epoch",
    "first_below_lcs_elapsed_seconds",
    "best_epoch",
    "best_val_metric_pct",

    "epochs_without_improvement",

    "train_sigma11_pct",
    "train_sigma22_pct",
    "train_sigma33_pct",
    "train_sigma12_pct",
    "train_sigma13_pct",
    "train_sigma23_pct",

    "val_sigma11_pct",
    "val_sigma22_pct",
    "val_sigma33_pct",
    "val_sigma12_pct",
    "val_sigma13_pct",
    "val_sigma23_pct",
)


def _now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _format_seconds(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "-"
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return "%d:%02d:%02d" % (hours, minutes, secs)
    return "%02d:%02d" % (minutes, secs)


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_history_csv(
    path: Path,
    history: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=HISTORY_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in history:
            writer.writerow(
                {name: row.get(name, "") for name in HISTORY_FIELDS}
            )

    os.replace(temporary, path)


def _torch_load_checkpoint(path: Path, map_location):
    # PyTorch 2.6 changed the default of weights_only. Training checkpoints
    # contain optimizer/RNG/history objects and require weights_only=False.
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _current_learning_rate(
    optimizer: torch.optim.Optimizer,
) -> float:
    if not optimizer.param_groups:
        return float("nan")
    return float(optimizer.param_groups[0]["lr"])


def _config_signature(cfg=CFG) -> Dict[str, object]:
    """
    Fields that must not change when resuming the same experiment.

    MAX_EPOCHS and stopping patience are intentionally excluded so an existing
    run can be extended. Settings that change the model, data presented in one
    optimizer step, objective, normalization, optimizer or scheduler are kept
    strict because changing them would no longer be an exact continuation.
    """
    return {
        "phase": str(cfg.PHASE),
        "multiplier": int(cfg.MULTIPLIER),
        "model_type": str(cfg.MODEL_TYPE),
        "dataset_run_name": str(cfg.DATASET_RUN_NAME),
        "h5_data_path": str(Path(cfg.H5_DATA_PATH).resolve()),
        "normalization_path": str(Path(cfg.NORMALIZATION_PATH).resolve()),
        "input_dim": int(_config_module.INPUT_DIM),
        "target_dim": int(_config_module.TARGET_DIM),

        "random_seed": int(cfg.RANDOM_SEED),
        "deterministic_torch": bool(cfg.DETERMINISTIC_TORCH),
        "batch_size": int(cfg.BATCH_SIZE),
        "num_workers": int(cfg.NUM_WORKERS),
        "drop_last_train_batch": bool(cfg.DROP_LAST_TRAIN_BATCH),
        "max_train_trajectories": cfg.MAX_TRAIN_TRAJECTORIES,
        "max_val_trajectories": cfg.MAX_VAL_TRAJECTORIES,

        "normalize_inputs": bool(cfg.NORMALIZE_INPUTS),
        "normalize_targets": bool(cfg.NORMALIZE_TARGETS),
        "normalization_mode": str(cfg.NORMALIZATION_MODE),
        "normalization_eps": float(cfg.NORMALIZATION_EPS),
        "use_sample_weight": bool(cfg.USE_SAMPLE_WEIGHT),

        "lstm_hidden_size": int(cfg.LSTM_HIDDEN_SIZE),
        "lstm_num_layers": int(cfg.LSTM_NUM_LAYERS),
        "lstm_dropout": float(cfg.LSTM_DROPOUT),
        "lstm_bidirectional": bool(cfg.LSTM_BIDIRECTIONAL),

        "mlp_hidden_size_1": int(cfg.MLP_HIDDEN_SIZE_1),
        "mlp_hidden_size_2": int(cfg.MLP_HIDDEN_SIZE_2),
        "mlp_activation": str(cfg.MLP_ACTIVATION),
        "mlp_dropout": float(cfg.MLP_DROPOUT),
        "mlp_weight_decay": float(cfg.MLP_WEIGHT_DECAY),

        "model_architecture_tag": str(
            cfg.MODEL_ARCHITECTURE_TAG
        ),
        "output_head": (
            "two_hidden_layer_mlp_plus_linear_output"
            if str(cfg.MODEL_TYPE) == "mlp"
            else "single_linear_output_layer"
        ),

        "linear_fit_intercept": bool(cfg.LINEAR_FIT_INTERCEPT),
        "ridge_fit_intercept": bool(cfg.RIDGE_FIT_INTERCEPT),
        "lasso_fit_intercept": bool(cfg.LASSO_FIT_INTERCEPT),
        "ridge_alpha": float(cfg.RIDGE_ALPHA),
        "lasso_alpha": float(cfg.LASSO_ALPHA),

        "optimizer": str(cfg.OPTIMIZER),
        "learning_rate": float(cfg.LEARNING_RATE),
        "weight_decay": float(cfg.WEIGHT_DECAY),
        "lr_scheduler": str(cfg.LR_SCHEDULER),
        "lr_reduce_factor": float(cfg.LR_REDUCE_FACTOR),
        "lr_reduce_patience": int(cfg.LR_REDUCE_PATIENCE),
        "min_learning_rate": float(cfg.MIN_LEARNING_RATE),
        "gradient_clip_norm": float(cfg.GRADIENT_CLIP_NORM),
        "use_amp": bool(cfg.USE_AMP),
        "data_loss": str(cfg.DATA_LOSS),

        "validation_metric": str(cfg.VALIDATION_METRIC),
        "exclude_first_step": bool(
            cfg.EXCLUDE_FIRST_STEP_FOR_LCS_COMPARISON
        ),
    }


def _validate_resume_signature(
    saved: Mapping[str, object],
    current: Mapping[str, object],
    strict: bool,
) -> None:
    differences = []
    keys = sorted(set(saved).union(current))
    for key in keys:
        if saved.get(key) != current.get(key):
            differences.append(
                "%s: saved=%r current=%r"
                % (key, saved.get(key), current.get(key))
            )

    if not differences:
        return

    message = (
        "Resume configuration differs from the checkpoint:\n  "
        + "\n  ".join(differences)
    )
    if strict:
        raise ValueError(message)
    print("[RESUME WARNING] " + message)


# =============================================================================
# 2. AMP compatibility
# =============================================================================

def _make_grad_scaler(
    enabled: bool,
):
    if not enabled:
        try:
            return torch.amp.GradScaler("cuda", enabled=False)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=False)

    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def _autocast_context(
    device: torch.device,
    enabled: bool,
):
    if not enabled:
        return contextlib.nullcontext()

    if device.type != "cuda":
        return contextlib.nullcontext()

    try:
        return torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )
    except Exception:
        return torch.cuda.amp.autocast(
            dtype=torch.float16,
            enabled=True,
        )


# =============================================================================
# 3. Optimizer and scheduler
# =============================================================================

def build_optimizer(
    model: nn.Module,
    cfg=CFG,
) -> torch.optim.Optimizer:
    model_type = str(cfg.MODEL_TYPE)

    # Ridge/Lasso penalties are explicit in 10_5. Classical regression
    # baselines do not use optimizer weight decay. LSTM and MLP have separate
    # neural-network weight-decay settings.
    if model_type == "lstm_fc":
        weight_decay = float(cfg.WEIGHT_DECAY)
    elif model_type == "mlp":
        weight_decay = float(cfg.MLP_WEIGHT_DECAY)
    else:
        weight_decay = 0.0

    kwargs = {
        "params": model.parameters(),
        "lr": float(cfg.LEARNING_RATE),
        "weight_decay": weight_decay,
    }

    if cfg.OPTIMIZER == "adam":
        return torch.optim.Adam(**kwargs)
    if cfg.OPTIMIZER == "adamw":
        return torch.optim.AdamW(**kwargs)

    raise ValueError(
        "Unsupported optimizer=%r" % cfg.OPTIMIZER
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg=CFG,
):
    if cfg.LR_SCHEDULER == "none":
        return None

    if cfg.LR_SCHEDULER == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg.LR_REDUCE_FACTOR),
            patience=int(cfg.LR_REDUCE_PATIENCE),
            min_lr=float(cfg.MIN_LEARNING_RATE),
        )

    raise ValueError(
        "Unsupported LR_SCHEDULER=%r" % cfg.LR_SCHEDULER
    )


# =============================================================================
# 4. RNG checkpointing
# =============================================================================

def capture_rng_state(
    train_generator: torch.Generator,
) -> Dict[str, object]:
    payload = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "train_generator_state": train_generator.get_state(),
    }

    if torch.cuda.is_available():
        payload["torch_cuda_rng_state_all"] = (
            torch.cuda.get_rng_state_all()
        )
    else:
        payload["torch_cuda_rng_state_all"] = None

    return payload


def _as_cpu_byte_tensor(
    value: object,
    state_name: str,
) -> torch.Tensor:
    """
    Convert a checkpoint RNG state to the CPU uint8 tensor required by
    PyTorch's RNG APIs.

    Checkpoints were previously loaded with map_location='cuda'. That also
    moved the saved CPU RNG tensor and DataLoader-generator tensor to CUDA,
    while torch.set_rng_state() and Generator.set_state() require CPU
    ByteTensor objects.
    """
    if torch.is_tensor(value):
        tensor = value.detach().to(
            device="cpu",
            dtype=torch.uint8,
        ).contiguous()
    else:
        try:
            tensor = torch.as_tensor(
                value,
                dtype=torch.uint8,
                device="cpu",
            ).contiguous()
        except Exception as exc:
            raise TypeError(
                "Checkpoint %s cannot be converted to a CPU ByteTensor; "
                "got %s"
                % (state_name, type(value).__name__)
            ) from exc

    if tensor.ndim != 1:
        tensor = tensor.reshape(-1).contiguous()

    if tensor.numel() == 0:
        raise ValueError(
            "Checkpoint %s is empty" % state_name
        )

    return tensor


def restore_rng_state(
    state: Mapping[str, object],
    train_generator: torch.Generator,
) -> None:
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])

    torch_cpu_state = _as_cpu_byte_tensor(
        state["torch_cpu_rng_state"],
        "torch_cpu_rng_state",
    )
    torch.set_rng_state(torch_cpu_state)

    train_generator_state = _as_cpu_byte_tensor(
        state["train_generator_state"],
        "train_generator_state",
    )
    train_generator.set_state(
        train_generator_state
    )

    cuda_state = state.get("torch_cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        if not isinstance(cuda_state, (list, tuple)):
            raise TypeError(
                "torch_cuda_rng_state_all must be a list/tuple, got %s"
                % type(cuda_state).__name__
            )

        cuda_states = [
            _as_cpu_byte_tensor(
                item,
                "torch_cuda_rng_state_all[%d]" % index,
            )
            for index, item in enumerate(cuda_state)
        ]

        current_device_count = torch.cuda.device_count()
        if len(cuda_states) != current_device_count:
            raise ValueError(
                "Checkpoint stores %d CUDA RNG states, but the current "
                "runtime exposes %d CUDA devices"
                % (len(cuda_states), current_device_count)
            )

        torch.cuda.set_rng_state_all(cuda_states)


# =============================================================================
# 5. Epoch accumulation
# =============================================================================

@dataclass
class EpochResult:
    total_loss: float
    data_loss: float
    regularization_loss: float
    metric_pct: float
    component_metric_pct: Tuple[float, ...]
    auxiliary_metrics: Dict[str, object]


class NormalizedDataLossAccumulator:
    """
    Exact epoch-wide weighted MSE in normalized target space.
    """

    def __init__(self) -> None:
        self.numerator = 0.0
        self.denominator = 0.0

    @torch.no_grad()
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes differ")

        weight = weight.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )

        if prediction.ndim == 3:
            if weight.ndim != 1:
                weight = weight.reshape(-1)
            expanded = weight[:, None, None]
            denominator = (
                float(torch.sum(weight).item())
                * prediction.shape[1]
                * prediction.shape[2]
            )
        elif prediction.ndim == 2:
            if weight.ndim != 1:
                weight = weight.reshape(-1)
            expanded = weight[:, None]
            denominator = (
                float(torch.sum(weight).item())
                * prediction.shape[1]
            )
        else:
            raise ValueError(
                "Prediction must be rank 2 or 3"
            )

        numerator = torch.sum(
            expanded * (prediction - target).pow(2)
        )

        self.numerator += float(numerator.item())
        self.denominator += float(denominator)

    def compute(self) -> float:
        if self.denominator <= 0.0:
            raise RuntimeError(
                "No normalized data loss observations"
            )
        return self.numerator / self.denominator


def _move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"

    x = batch["x"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=non_blocking,
    )
    y = batch["y"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=non_blocking,
    )
    weight = batch["weight"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=non_blocking,
    )

    return x, y, weight


def run_one_epoch(
    model: nn.Module,
    loader,
    torch_normalizer,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler,
    cfg=CFG,
) -> EpochResult:
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    data_accumulator = NormalizedDataLossAccumulator()
    metric_accumulator = StressMetricAccumulator(
        exclude_first_step=bool(
            cfg.EXCLUDE_FIRST_STEP_FOR_LCS_COMPARISON
        )
    )

    for batch in loader:
        x_physical, y_physical, weight = (
            _move_batch_to_device(batch, device)
        )

        # USE_SAMPLE_WEIGHT controls both the optimization objective and every
        # reported metric. Replacing weights with ones preserves the same code
        # path while making the result exactly unweighted.
        if not bool(cfg.USE_SAMPLE_WEIGHT):
            weight = torch.ones_like(weight)

        x_normalized = (
            torch_normalizer.normalize_input(x_physical)
            if cfg.NORMALIZE_INPUTS
            else x_physical
        )
        y_normalized = (
            torch_normalizer.normalize_target(y_physical)
            if cfg.NORMALIZE_TARGETS
            else y_physical
        )

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with _autocast_context(
                device=device,
                enabled=bool(
                    cfg.USE_AMP
                    and device.type == "cuda"
                ),
            ):
                prediction_normalized = model(x_normalized)
                loss_breakdown = compute_training_loss(
                    prediction_normalized=prediction_normalized,
                    target_normalized=y_normalized,
                    sample_weight=weight,
                    model=model,
                    cfg=cfg,
                )

            if is_training:
                scaler.scale(
                    loss_breakdown.total_loss
                ).backward()

                scaler.unscale_(optimizer)

                if float(cfg.GRADIENT_CLIP_NORM) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(
                            cfg.GRADIENT_CLIP_NORM
                        ),
                    )

                scaler.step(optimizer)
                scaler.update()

        with torch.no_grad():
            data_accumulator.update(
                prediction=prediction_normalized.detach(),
                target=y_normalized.detach(),
                weight=weight,
            )

            if bool(cfg.NORMALIZE_TARGETS):
                update_physical_metric_from_normalized(
                    accumulator=metric_accumulator,
                    prediction_normalized=prediction_normalized.detach(),
                    target_normalized=y_normalized.detach(),
                    sample_weight=weight,
                    torch_normalizer=torch_normalizer,
                )
            else:
                # The model and target are already in physical stress units.
                # Calling the denormalization helper here would incorrectly
                # multiply by target_std and add target_mean a second time.
                metric_accumulator.update(
                    prediction_physical=prediction_normalized.detach(),
                    target_physical=y_physical.detach(),
                    sample_weight=weight,
                )

        del (
            x_physical,
            y_physical,
            weight,
            x_normalized,
            y_normalized,
            prediction_normalized,
            loss_breakdown,
        )

    data_loss = float(data_accumulator.compute())

    with torch.no_grad():
        regularization = float(
            regression_regularization_penalty(
                model=model,
                cfg=cfg,
            ).item()
        )

    metrics = metric_accumulator.compute()
    main_metric = float(
        metrics[cfg.VALIDATION_METRIC]
    )
    component_values = tuple(
        float(x)
        for x in metrics[
            "weighted_component_relative_l2_pct"
        ]
    )

    return EpochResult(
        total_loss=data_loss + regularization,
        data_loss=data_loss,
        regularization_loss=regularization,
        metric_pct=main_metric,
        component_metric_pct=component_values,
        auxiliary_metrics=metrics,
    )


# =============================================================================
# 6. Dashboard and live table
# =============================================================================

class LiveDashboard:
    """
    The dashboard is reconstructed from the complete history every time.
    Resumed training therefore restores the original full curves.
    """

    def __init__(
        self,
        cfg,
        lcs_line_pct: float,
    ) -> None:
        self.cfg = cfg
        self.lcs_line_pct = float(lcs_line_pct)
        self.figure = None
        self.axis = None
        self.pyplot = None
        self.window_enabled = False

        if not cfg.ENABLE_LIVE_DASHBOARD:
            return

        try:
            import matplotlib.pyplot as plt

            self.pyplot = plt
            backend = str(plt.get_backend()).lower()
            self.window_enabled = bool(
                cfg.DASHBOARD_SHOW_WINDOW
                and "agg" not in backend
            )

            if self.window_enabled:
                plt.ion()

            self.figure, self.axis = plt.subplots(
                figsize=(9.5, 6.0)
            )
        except Exception as exc:
            print(
                "[DASHBOARD WARNING] Plot initialization failed: %s"
                % exc
            )
            self.pyplot = None
            self.figure = None
            self.axis = None
            self.window_enabled = False

    def render(
        self,
        history: Sequence[Mapping[str, object]],
        force_save: bool = False,
    ) -> None:
        if not history:
            return

        self._render_console_table(history)

        if self.axis is None or self.figure is None:
            return

        epochs = np.asarray(
            [int(row["epoch"]) for row in history],
            dtype=np.int64,
        )
        train_metric = np.asarray(
            [float(row["train_metric_pct"]) for row in history],
            dtype=np.float64,
        )
        val_metric = np.asarray(
            [float(row["val_metric_pct"]) for row in history],
            dtype=np.float64,
        )

        self.axis.clear()
        self.axis.plot(
            epochs,
            train_metric,
            linewidth=1.8,
            label="Train error",
        )
        self.axis.plot(
            epochs,
            val_metric,
            linewidth=2.0,
            label="Validation error",
        )
        self.axis.axhline(
            self.lcs_line_pct,
            linestyle="--",
            linewidth=1.8,
            label="LCS baseline",
        )

        below_rows = [
            row
            for row in history
            if bool(row.get("is_below_lcs", False))
        ]
        if below_rows:
            first = below_rows[0]
            self.axis.scatter(
                [int(first["epoch"])],
                [float(first["val_metric_pct"])],
                s=45,
                zorder=5,
                label="First below LCS",
            )

        best_row = min(
            history,
            key=lambda row: float(
                row["val_metric_pct"]
            ),
        )
        self.axis.scatter(
            [int(best_row["epoch"])],
            [float(best_row["val_metric_pct"])],
            marker="*",
            s=100,
            zorder=6,
            label="Best validation",
        )

        if (
            self.cfg.DASHBOARD_Y_SCALE == "log"
            and np.all(train_metric > 0.0)
            and np.all(val_metric > 0.0)
            and self.lcs_line_pct > 0.0
        ):
            self.axis.set_yscale("log")
        else:
            self.axis.set_yscale("linear")

        elapsed = float(history[-1]["elapsed_seconds"])
        self.axis.set_title(
            "%s | %s | %dx | cumulative training %s"
            % (
                self.cfg.PHASE,
                self.cfg.model_display_name,
                self.cfg.MULTIPLIER,
                _format_seconds(elapsed),
            )
        )
        self.axis.set_xlabel("Epoch")
        self.axis.set_ylabel(
            "Weighted global relative L2 error (%)"
        )
        self.axis.grid(True, alpha=0.25)
        self.axis.legend(loc="best")
        self.figure.tight_layout()

        should_save = (
            force_save
            or (
                int(history[-1]["epoch"])
                % int(
                    self.cfg.SAVE_DASHBOARD_EVERY_EPOCHS
                )
                == 0
            )
        )

        if should_save:
            output = Path(
                self.cfg.LIVE_DASHBOARD_PATH
            )
            output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            temporary = output.with_suffix(
                output.suffix + ".tmp"
            )
            self.figure.savefig(
                temporary,
                dpi=int(self.cfg.DASHBOARD_DPI),
                format="png",
                bbox_inches="tight",
            )
            os.replace(temporary, output)

        if self.window_enabled:
            try:
                self.figure.canvas.draw_idle()
                self.figure.canvas.flush_events()
                self.pyplot.pause(0.001)
            except Exception as exc:
                print(
                    "[DASHBOARD WARNING] Interactive refresh failed: %s"
                    % exc
                )
                self.window_enabled = False

    def _render_console_table(
        self,
        history: Sequence[Mapping[str, object]],
    ) -> None:
        if (
            self.cfg.CLEAR_TERMINAL_FOR_LIVE_TABLE
            and sys.stdout.isatty()
        ):
            os.system("cls" if os.name == "nt" else "clear")

        last = history[-1]
        first_below_epoch = last.get(
            "first_below_lcs_epoch"
        )
        first_below_time = last.get(
            "first_below_lcs_elapsed_seconds"
        )

        print("=" * 132)
        print(
            "LIVE TRAINING | %s | %s | multiplier %dx | resumed history included"
            % (
                self.cfg.PHASE,
                self.cfg.model_display_name,
                self.cfg.MULTIPLIER,
            )
        )
        print(
            "LCS line: %.6f %% | Best val: %.6f %% @ epoch %s | "
            "First below LCS: epoch %s, time %s"
            % (
                self.lcs_line_pct,
                float(last["best_val_metric_pct"]),
                str(last.get("best_epoch", "-")),
                str(first_below_epoch)
                if first_below_epoch is not None
                else "-",
                _format_seconds(
                    first_below_time
                    if first_below_time is not None
                    else None
                ),
            )
        )
        print("-" * 132)
        print(
            "%6s %10s %10s %12s %12s %12s %12s %10s %8s"
            % (
                "Epoch",
                "Elapsed",
                "LR",
                "TrainLoss",
                "TrainErr%",
                "ValLoss",
                "ValErr%",
                "Val-LCS",
                "Below",
            )
        )
        print("-" * 132)

        rows = history[-int(self.cfg.LIVE_TABLE_ROWS):]
        for row in rows:
            print(
                "%6d %10s %10.3e %12.5e %12.5f %12.5e %12.5f %10.5f %8s"
                % (
                    int(row["epoch"]),
                    _format_seconds(
                        float(row["elapsed_seconds"])
                    ),
                    float(row["learning_rate"]),
                    float(row["train_total_loss"]),
                    float(row["train_metric_pct"]),
                    float(row["val_total_loss"]),
                    float(row["val_metric_pct"]),
                    float(
                        row["val_minus_lcs_pct_point"]
                    ),
                    "YES"
                    if bool(row["is_below_lcs"])
                    else "NO",
                )
            )

        print("=" * 132)
        print(
            "Full history CSV : %s"
            % self.cfg.HISTORY_CSV_PATH
        )
        print(
            "Full history plot: %s"
            % self.cfg.LIVE_DASHBOARD_PATH
        )

    def close(self) -> None:
        if self.pyplot is not None and self.figure is not None:
            try:
                self.pyplot.close(self.figure)
            except Exception:
                pass


# =============================================================================
# 7. Trainer state and checkpointing
# =============================================================================

@dataclass
class ResumeState:
    start_epoch: int
    cumulative_elapsed_seconds: float
    history: List[Dict[str, object]]
    crossing_tracker: LCSCrossingTracker
    best_val_metric_pct: float
    best_epoch: Optional[int]
    epochs_without_improvement: int
    resumed_from: Optional[Path]


class Trainer:
    def __init__(self, cfg=CFG) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)

        cfg.validate_required_files()
        cfg.create_output_directories()
        cfg.save_snapshot()

        self.bundle: LSTMDataBundle = (
            build_lstm_data_bundle(cfg)
        )

        self.normalization_stats = (
            compile_train_only_normalization(cfg)
        )
        self.torch_normalizer = TorchNormalizer(
            self.normalization_stats,
            device=self.device,
        )

        self.model = build_model(cfg)
        self.model_info = get_model_info(
            self.model,
            cfg,
        )
        save_model_structure(self.model, cfg)
        save_metric_definition(
            Path(cfg.EXPERIMENT_ROOT)
            / "loss_and_metric_definition.json",
            cfg,
        )

        self.optimizer = build_optimizer(
            self.model,
            cfg,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            cfg,
        )

        amp_enabled = bool(
            cfg.USE_AMP
            and self.device.type == "cuda"
        )
        self.scaler = _make_grad_scaler(
            enabled=amp_enabled
        )

        self.lcs_line_pct = float(
            self.bundle
            .lcs_reference
            .horizontal_line_pct
        )

        self.dashboard = LiveDashboard(
            cfg=cfg,
            lcs_line_pct=self.lcs_line_pct,
        )

        self.resume_state = self._initialize_or_resume()

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def _resolve_resume_path(self) -> Optional[Path]:
        custom = str(
            self.cfg.RESUME_CHECKPOINT_PATH
        ).strip()

        if custom:
            path = Path(custom)
        else:
            path = Path(
                self.cfg.LAST_CHECKPOINT_PATH
            )

        exists = path.is_file()
        mode = str(self.cfg.RESUME_MODE)

        if mode == "never":
            if exists:
                raise FileExistsError(
                    "RESUME_MODE='never', but a last checkpoint already "
                    "exists. Use RESUME_MODE='auto', change the experiment "
                    "identity, or remove the old run after backing it up: %s"
                    % path
                )
            return None

        if mode == "required":
            if not exists:
                raise FileNotFoundError(
                    "RESUME_MODE='required', but checkpoint is absent: %s"
                    % path
                )
            return path

        if mode == "auto":
            return path if exists else None

        raise ValueError(
            "Unsupported RESUME_MODE=%r" % mode
        )

    def _initialize_or_resume(self) -> ResumeState:
        resume_path = self._resolve_resume_path()

        if resume_path is None:
            tracker = LCSCrossingTracker(
                lcs_line_pct=self.lcs_line_pct
            )
            return ResumeState(
                start_epoch=1,
                cumulative_elapsed_seconds=0.0,
                history=[],
                crossing_tracker=tracker,
                best_val_metric_pct=float("inf"),
                best_epoch=None,
                epochs_without_improvement=0,
                resumed_from=None,
            )

        # Always deserialize checkpoints on CPU. Model and optimizer states
        # are moved to their parameter devices by load_state_dict(), while RNG
        # state APIs specifically require CPU ByteTensor objects.
        checkpoint = _torch_load_checkpoint(
            resume_path,
            map_location="cpu",
        )

        if checkpoint.get("schema_name") != (
            "CPFE_STEP10_TRAINING_CHECKPOINT_V1"
        ):
            raise ValueError(
                "Unsupported checkpoint schema in %s"
                % resume_path
            )

        _validate_resume_signature(
            saved=checkpoint["config_signature"],
            current=_config_signature(self.cfg),
            strict=bool(
                self.cfg.STRICT_RESUME_CONFIG
            ),
        )

        saved_lcs = float(
            checkpoint["lcs_line_pct"]
        )
        if not math.isclose(
            saved_lcs,
            self.lcs_line_pct,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Saved LCS line %.12g differs from current %.12g"
                % (saved_lcs, self.lcs_line_pct)
            )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        saved_scheduler = checkpoint.get(
            "scheduler_state_dict"
        )
        if self.scheduler is not None:
            if saved_scheduler is None:
                raise ValueError(
                    "Current run uses a scheduler, but checkpoint does not"
                )
            self.scheduler.load_state_dict(
                saved_scheduler
            )
        elif saved_scheduler is not None:
            raise ValueError(
                "Checkpoint uses a scheduler, but the current run does not"
            )

        scaler_state = checkpoint.get(
            "scaler_state_dict"
        )
        if scaler_state is not None:
            self.scaler.load_state_dict(
                scaler_state
            )

        restore_rng_state(
            checkpoint["rng_state"],
            self.bundle.train_generator,
        )

        history = [
            dict(row)
            for row in checkpoint["history"]
        ]

        completed_epoch = int(
            checkpoint["completed_epoch"]
        )
        if len(history) != completed_epoch:
            raise ValueError(
                "Checkpoint history length=%d, completed_epoch=%d"
                % (len(history), completed_epoch)
            )
        if history and int(history[-1]["epoch"]) != completed_epoch:
            raise ValueError(
                "Last history epoch does not match completed_epoch"
            )

        tracker = LCSCrossingTracker.from_state_dict(
            checkpoint["crossing_tracker_state"]
        )

        state = ResumeState(
            start_epoch=completed_epoch + 1,
            cumulative_elapsed_seconds=float(
                checkpoint[
                    "cumulative_elapsed_seconds"
                ]
            ),
            history=history,
            crossing_tracker=tracker,
            best_val_metric_pct=float(
                checkpoint["best_val_metric_pct"]
            ),
            best_epoch=(
                None
                if checkpoint.get("best_epoch") is None
                else int(checkpoint["best_epoch"])
            ),
            epochs_without_improvement=int(
                checkpoint[
                    "epochs_without_improvement"
                ]
            ),
            resumed_from=resume_path,
        )

        # Rebuild persistent artifacts immediately. This guarantees that a
        # deleted/stale CSV or PNG is restored from checkpoint history.
        _atomic_write_history_csv(
            Path(self.cfg.HISTORY_CSV_PATH),
            history,
        )
        if history:
            self.dashboard.render(
                history,
                force_save=True,
            )

        print("=" * 108)
        print("Training resumed")
        print("Checkpoint          :", resume_path)
        print("Completed epoch     :", completed_epoch)
        print("Next epoch          :", state.start_epoch)
        print(
            "Cumulative time     :",
            _format_seconds(
                state.cumulative_elapsed_seconds
            ),
        )
        print("History rows        :", len(history))
        print("Best val error      :", state.best_val_metric_pct)
        print(
            "First below LCS     :",
            tracker.first_below_epoch,
        )
        print("=" * 108)

        return state

    # ------------------------------------------------------------------
    # Checkpoint payload
    # ------------------------------------------------------------------

    def _checkpoint_payload(
        self,
        completed_epoch: int,
        cumulative_elapsed_seconds: float,
        history: Sequence[Mapping[str, object]],
        tracker: LCSCrossingTracker,
        best_val_metric_pct: float,
        best_epoch: Optional[int],
        epochs_without_improvement: int,
    ) -> Dict[str, object]:
        return {
            "schema_name":
                "CPFE_STEP10_TRAINING_CHECKPOINT_V1",
            "saved_time": _now_string(),
            "completed_epoch": int(completed_epoch),
            "cumulative_elapsed_seconds": float(
                cumulative_elapsed_seconds
            ),

            "phase": str(self.cfg.PHASE),
            "multiplier": int(self.cfg.MULTIPLIER),
            "model_type": str(self.cfg.MODEL_TYPE),
            "model_info": self.model_info.to_dict(),
            "config_signature":
                _config_signature(self.cfg),
            "lcs_line_pct": float(
                self.lcs_line_pct
            ),

            "model_state_dict":
                self.model.state_dict(),
            "optimizer_state_dict":
                self.optimizer.state_dict(),
            "scheduler_state_dict":
                None
                if self.scheduler is None
                else self.scheduler.state_dict(),
            "scaler_state_dict":
                self.scaler.state_dict(),

            "history": [
                dict(row)
                for row in history
            ],
            "crossing_tracker_state":
                tracker.state_dict(),
            "best_val_metric_pct":
                float(best_val_metric_pct),
            "best_epoch":
                best_epoch,
            "epochs_without_improvement":
                int(epochs_without_improvement),

            "rng_state": capture_rng_state(
                self.bundle.train_generator
            ),
        }

    def _save_completed_epoch_state(
        self,
        completed_epoch: int,
        cumulative_elapsed_seconds: float,
        history: Sequence[Mapping[str, object]],
        tracker: LCSCrossingTracker,
        best_val_metric_pct: float,
        best_epoch: Optional[int],
        epochs_without_improvement: int,
        is_new_best: bool,
    ) -> None:
        payload = self._checkpoint_payload(
            completed_epoch=completed_epoch,
            cumulative_elapsed_seconds=(
                cumulative_elapsed_seconds
            ),
            history=history,
            tracker=tracker,
            best_val_metric_pct=(
                best_val_metric_pct
            ),
            best_epoch=best_epoch,
            epochs_without_improvement=(
                epochs_without_improvement
            ),
        )

        # Save the resume checkpoint first. CSV/figure are then rebuildable
        # from its embedded complete history.
        if self.cfg.SAVE_LAST_CHECKPOINT:
            _atomic_torch_save(
                Path(self.cfg.LAST_CHECKPOINT_PATH),
                payload,
            )

        if (
            is_new_best
            and self.cfg.SAVE_BEST_CHECKPOINT
        ):
            _atomic_torch_save(
                Path(self.cfg.BEST_CHECKPOINT_PATH),
                payload,
            )

        every = int(self.cfg.SAVE_EVERY_N_EPOCHS)
        if every > 0 and completed_epoch % every == 0:
            periodic = (
                Path(self.cfg.CHECKPOINT_DIR)
                / (
                    "epoch_%06d.pt"
                    % completed_epoch
                )
            )
            _atomic_torch_save(
                periodic,
                payload,
            )

        _atomic_write_history_csv(
            Path(self.cfg.HISTORY_CSV_PATH),
            history,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, object]:
        state = self.resume_state
        history = state.history
        tracker = state.crossing_tracker

        if state.start_epoch > int(
            self.cfg.MAX_EPOCHS
        ):
            print(
                "MAX_EPOCHS=%d is not greater than completed epoch %d. "
                "Nothing to continue."
                % (
                    self.cfg.MAX_EPOCHS,
                    state.start_epoch - 1,
                )
            )
            if history:
                self.dashboard.render(
                    history,
                    force_save=True,
                )
            return self._build_final_summary(
                history=history,
                tracker=tracker,
                stopped_reason="already_complete",
            )

        best_val = float(
            state.best_val_metric_pct
        )
        best_epoch = state.best_epoch
        epochs_without_improvement = int(
            state.epochs_without_improvement
        )

        session_started = time.perf_counter()
        stopped_reason = "max_epochs"

        try:
            for epoch in range(
                state.start_epoch,
                int(self.cfg.MAX_EPOCHS) + 1,
            ):
                epoch_started = time.perf_counter()
                # Record the rate actually used during this epoch. A plateau
                # scheduler is stepped after validation and changes the rate for
                # the following epoch.
                epoch_learning_rate = _current_learning_rate(
                    self.optimizer
                )

                train_result = run_one_epoch(
                    model=self.model,
                    loader=self.bundle.train_loader,
                    torch_normalizer=(
                        self.torch_normalizer
                    ),
                    device=self.device,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    cfg=self.cfg,
                )

                val_result = run_one_epoch(
                    model=self.model,
                    loader=self.bundle.val_loader,
                    torch_normalizer=(
                        self.torch_normalizer
                    ),
                    device=self.device,
                    optimizer=None,
                    scaler=self.scaler,
                    cfg=self.cfg,
                )

                epoch_seconds = (
                    time.perf_counter()
                    - epoch_started
                )
                cumulative_elapsed = (
                    state.cumulative_elapsed_seconds
                    + (
                        time.perf_counter()
                        - session_started
                    )
                )

                crossing_update = tracker.update(
                    epoch=epoch,
                    elapsed_seconds=cumulative_elapsed,
                    validation_metric_pct=(
                        val_result.metric_pct
                    ),
                )

                min_delta = float(
                    self.cfg.EARLY_STOPPING_MIN_DELTA
                )
                is_new_best = (
                    val_result.metric_pct
                    < best_val - min_delta
                )

                if is_new_best:
                    best_val = float(
                        val_result.metric_pct
                    )
                    best_epoch = int(epoch)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if self.scheduler is not None:
                    self.scheduler.step(
                        val_result.metric_pct
                    )

                record = build_epoch_record(
                    epoch=epoch,
                    elapsed_seconds=(
                        cumulative_elapsed
                    ),
                    learning_rate=epoch_learning_rate,
                    train_total_loss=(
                        train_result.total_loss
                    ),
                    train_data_loss=(
                        train_result.data_loss
                    ),
                    train_regularization_loss=(
                        train_result
                        .regularization_loss
                    ),
                    train_metric_pct=(
                        train_result.metric_pct
                    ),
                    val_total_loss=(
                        val_result.total_loss
                    ),
                    val_data_loss=(
                        val_result.data_loss
                    ),
                    val_regularization_loss=(
                        val_result
                        .regularization_loss
                    ),
                    val_metric_pct=(
                        val_result.metric_pct
                    ),
                    lcs_line_pct=(
                        self.lcs_line_pct
                    ),
                    crossing_update=(
                        crossing_update
                    ),
                )
                record["epoch_seconds"] = float(
                    epoch_seconds
                )
                record[
                    "epochs_without_improvement"
                ] = int(
                    epochs_without_improvement
                )

                for name, value in zip(
                    TARGET_ORDER,
                    train_result.component_metric_pct,
                ):
                    record[
                        "train_%s_pct" % name
                    ] = float(value)

                for name, value in zip(
                    TARGET_ORDER,
                    val_result.component_metric_pct,
                ):
                    record[
                        "val_%s_pct" % name
                    ] = float(value)

                history.append(record)

                self._save_completed_epoch_state(
                    completed_epoch=epoch,
                    cumulative_elapsed_seconds=(
                        cumulative_elapsed
                    ),
                    history=history,
                    tracker=tracker,
                    best_val_metric_pct=(
                        best_val
                    ),
                    best_epoch=best_epoch,
                    epochs_without_improvement=(
                        epochs_without_improvement
                    ),
                    is_new_best=is_new_best,
                )

                if (
                    epoch
                    % int(
                        self.cfg
                        .DASHBOARD_REFRESH_EVERY_EPOCHS
                    )
                    == 0
                ):
                    self.dashboard.render(
                        history,
                        force_save=False,
                    )

                if (
                    crossing_update[
                        "just_crossed_lcs"
                    ]
                ):
                    print(
                        "[LCS CROSSED] epoch=%d, val=%.6f%%, "
                        "LCS=%.6f%%, cumulative time=%s"
                        % (
                            epoch,
                            val_result.metric_pct,
                            self.lcs_line_pct,
                            _format_seconds(
                                cumulative_elapsed
                            ),
                        )
                    )

                if (
                    epochs_without_improvement
                    >= int(
                        self.cfg
                        .EARLY_STOPPING_PATIENCE
                    )
                ):
                    stopped_reason = "early_stopping"
                    print(
                        "Early stopping at epoch %d: no improvement "
                        "for %d epochs."
                        % (
                            epoch,
                            epochs_without_improvement,
                        )
                    )
                    break

        except KeyboardInterrupt:
            stopped_reason = "keyboard_interrupt"
            print(
                "\n[INTERRUPTED] Current unfinished epoch is discarded. "
                "The next run will resume from the last completed epoch."
            )
            if self.cfg.SAVE_INTERRUPTION_NOTE:
                _atomic_write_json(
                    Path(self.cfg.EXPERIMENT_ROOT)
                    / "interruption_note.json",
                    {
                        "time": _now_string(),
                        "message": (
                            "Interrupted inside an epoch. "
                            "Auto-resume uses last_checkpoint.pt, "
                            "which contains the complete history through "
                            "the previous finished epoch."
                        ),
                        "last_completed_epoch": (
                            int(history[-1]["epoch"])
                            if history
                            else 0
                        ),
                        "history_rows": len(history),
                        "last_checkpoint": (
                            self.cfg.LAST_CHECKPOINT_PATH
                        ),
                    },
                )
            if history:
                _atomic_write_history_csv(
                    Path(
                        self.cfg.HISTORY_CSV_PATH
                    ),
                    history,
                )
                self.dashboard.render(
                    history,
                    force_save=True,
                )

        except Exception:
            stopped_reason = "exception"
            error_path = (
                Path(self.cfg.EXPERIMENT_ROOT)
                / "training_failed.txt"
            )
            error_path.write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            if history:
                _atomic_write_history_csv(
                    Path(
                        self.cfg.HISTORY_CSV_PATH
                    ),
                    history,
                )
                self.dashboard.render(
                    history,
                    force_save=True,
                )
            raise

        finally:
            self.dashboard.close()
            self.bundle.close()

        summary = self._build_final_summary(
            history=history,
            tracker=tracker,
            stopped_reason=stopped_reason,
        )
        _atomic_write_json(
            Path(
                self.cfg.TRAINING_SUMMARY_PATH
            ),
            summary,
        )
        return summary

    def _build_final_summary(
        self,
        history: Sequence[Mapping[str, object]],
        tracker: LCSCrossingTracker,
        stopped_reason: str,
    ) -> Dict[str, object]:
        best_row = (
            min(
                history,
                key=lambda row: float(
                    row["val_metric_pct"]
                ),
            )
            if history
            else None
        )

        return {
            "schema_name":
                "CPFE_STEP10_TRAINING_SUMMARY_V1",
            "created_time": _now_string(),
            "phase": str(self.cfg.PHASE),
            "multiplier": int(
                self.cfg.MULTIPLIER
            ),
            "model_type": str(
                self.cfg.MODEL_TYPE
            ),
            "model_display_name":
                self.cfg.model_display_name,
            "stopped_reason": stopped_reason,
            "completed_epochs": len(history),
            "last_epoch": (
                int(history[-1]["epoch"])
                if history
                else 0
            ),
            "cumulative_training_seconds": (
                float(
                    history[-1][
                        "elapsed_seconds"
                    ]
                )
                if history
                else 0.0
            ),
            "lcs_line_pct": float(
                self.lcs_line_pct
            ),
            "first_below_lcs_epoch":
                tracker.first_below_epoch,
            "first_below_lcs_elapsed_seconds":
                tracker.first_below_elapsed_seconds,
            "first_below_lcs_metric_pct":
                tracker.first_below_metric_pct,
            "best_epoch": (
                None
                if best_row is None
                else int(best_row["epoch"])
            ),
            "best_val_metric_pct": (
                None
                if best_row is None
                else float(
                    best_row["val_metric_pct"]
                )
            ),
            "best_checkpoint":
                self.cfg.BEST_CHECKPOINT_PATH,
            "last_checkpoint":
                self.cfg.LAST_CHECKPOINT_PATH,
            "history_csv":
                self.cfg.HISTORY_CSV_PATH,
            "live_dashboard":
                self.cfg.LIVE_DASHBOARD_PATH,
            "resumed_from": (
                None
                if self.resume_state.resumed_from
                is None
                else str(
                    self.resume_state
                    .resumed_from
                )
            ),
        }


# =============================================================================
# 8. Public entry
# =============================================================================

def train_selected_model(cfg=CFG) -> Dict[str, object]:
    cfg.print_summary()
    trainer = Trainer(cfg)
    summary = trainer.train()

    print("=" * 108)
    print("Training finished")
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=" * 108)
    return summary


# =============================================================================
# 9. Lightweight standalone check
# =============================================================================

def smoke_test_checkpoint_helpers() -> None:
    tracker = LCSCrossingTracker(
        lcs_line_pct=5.0
    )
    tracker.update(
        epoch=1,
        elapsed_seconds=3.0,
        validation_metric_pct=6.0,
    )
    tracker.update(
        epoch=2,
        elapsed_seconds=7.0,
        validation_metric_pct=4.0,
    )

    state = tracker.state_dict()
    restored = (
        LCSCrossingTracker.from_state_dict(
            state
        )
    )

    if restored.first_below_epoch != 2:
        raise RuntimeError(
            "Crossing tracker resume test failed"
        )
    if restored.best_epoch != 2:
        raise RuntimeError(
            "Best-epoch resume test failed"
        )

    print("Trainer checkpoint-helper smoke test: PASS")


if __name__ == "__main__":
    # This file defines the trainer. The formal execution entry will be
    # 10_7_train.py. Running 10_6 directly only checks resume helpers and does
    # not start the large training job accidentally.
    CFG.print_summary()
    smoke_test_checkpoint_helpers()
    print(
        "Trainer definition is ready. Formal training will be started "
        "by 10_7_train.py."
    )
