#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_5_loss_and_metrics.py

Unified Step-10 losses, validation metrics and LCS comparison.

Training objective
------------------
All four models use data-only supervision. No physics-informed term is used.

Linear:
    total_loss = weighted_MSE

Ridge:
    total_loss = weighted_MSE + alpha * sum(W^2)

Lasso:
    total_loss = weighted_MSE + alpha * sum(abs(W))

LSTM + FC:
    total_loss = weighted_MSE

The weighted MSE is evaluated in normalized target space. Sample_Weight is a
trajectory/sample weight and is never a model input.

LCS-comparable validation metric
--------------------------------
The model validation curve and the Step-09 LCS horizontal line use exactly the
same physical-unit metric:

    100 * sqrt(
        sum(w * (sigma_pred - sigma_true)^2)
        /
        sum(w * sigma_true^2)
    )

For complete sequences, time index 0 is excluded because strict LCS has no
previous converged solution there.

This module also records:
- the first epoch/time at which validation error falls below LCS;
- the best validation metric;
- weighted and unweighted auxiliary metrics;
- per-stress-component relative-L2 errors.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

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
    "step10_config_for_loss",
    "10_1_config.py",
)
_models_module = _import_local_module(
    "step10_models_for_loss",
    "10_4_models.py",
)

CFG = _config_module.CFG
TARGET_DIM = int(_config_module.TARGET_DIM)
TARGET_ORDER = tuple(_config_module.TARGET_ORDER)

PointwiseLinearModel = _models_module.PointwiseLinearModel


# =============================================================================
# 1. Weight handling
# =============================================================================

def _expanded_weight_tensor(
    weight: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Expand sample/trajectory weights to the prediction tensor shape except for
    the final stress-component dimension.

    Supported
    ---------
    reference [B,L,C]:
        weight [B], [B,1], [B,L], [B,L,1]

    reference [P,C]:
        weight [P], [P,1]
    """
    if not torch.is_tensor(weight):
        weight = torch.as_tensor(
            weight,
            dtype=reference.dtype,
            device=reference.device,
        )
    else:
        weight = weight.to(
            device=reference.device,
            dtype=reference.dtype,
        )

    if reference.ndim == 3:
        batch, length, _ = reference.shape

        if weight.ndim == 1:
            if weight.shape[0] != batch:
                raise ValueError(
                    "Weight shape %s is incompatible with reference %s"
                    % (tuple(weight.shape), tuple(reference.shape))
                )
            weight = weight[:, None, None]

        elif weight.ndim == 2:
            if weight.shape == (batch, 1):
                weight = weight[:, :, None]
            elif weight.shape == (batch, length):
                weight = weight[:, :, None]
            else:
                raise ValueError(
                    "Weight shape %s is incompatible with reference %s"
                    % (tuple(weight.shape), tuple(reference.shape))
                )

        elif weight.ndim == 3:
            if weight.shape not in (
                (batch, 1, 1),
                (batch, length, 1),
            ):
                raise ValueError(
                    "Weight shape %s is incompatible with reference %s"
                    % (tuple(weight.shape), tuple(reference.shape))
                )
        else:
            raise ValueError(
                "Sequence weight rank must be 1, 2 or 3"
            )

        return weight

    if reference.ndim == 2:
        points, _ = reference.shape

        if weight.ndim == 1:
            if weight.shape[0] != points:
                raise ValueError(
                    "Weight shape %s is incompatible with reference %s"
                    % (tuple(weight.shape), tuple(reference.shape))
                )
            weight = weight[:, None]

        elif weight.ndim == 2:
            if weight.shape != (points, 1):
                raise ValueError(
                    "Weight shape %s is incompatible with reference %s"
                    % (tuple(weight.shape), tuple(reference.shape))
                )
        else:
            raise ValueError(
                "Pointwise weight rank must be 1 or 2"
            )

        return weight

    raise ValueError(
        "Reference must be [B,L,C] or [P,C], got %s"
        % (tuple(reference.shape),)
    )


def _validate_prediction_target(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction shape %s != target shape %s"
            % (tuple(prediction.shape), tuple(target.shape))
        )
    if prediction.ndim not in (2, 3):
        raise ValueError(
            "Prediction/target must be [P,C] or [B,L,C]"
        )
    if prediction.shape[-1] != TARGET_DIM:
        raise ValueError(
            "Stress dimension=%d, expected=%d"
            % (prediction.shape[-1], TARGET_DIM)
        )
    if not torch.is_floating_point(prediction):
        raise TypeError("Prediction must be floating point")
    if not torch.is_floating_point(target):
        raise TypeError("Target must be floating point")


# =============================================================================
# 2. Data loss and model regularization
# =============================================================================

def weighted_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted MSE over all stress components and selected time/point samples.

    For sequence data with one weight per trajectory:

        sum_b w_b * sum_t,c error^2
        --------------------------------
        sum_b w_b * L * C

    For pointwise data:

        sum_p w_p * sum_c error^2
        --------------------------
        sum_p w_p * C
    """
    _validate_prediction_target(prediction, target)

    weight = _expanded_weight_tensor(
        sample_weight,
        prediction,
    )

    if torch.any(~torch.isfinite(weight)):
        raise ValueError("Sample weights contain NaN/Inf")
    if torch.any(weight <= 0):
        raise ValueError("Sample weights must be positive")

    squared_error = (prediction - target).pow(2)
    numerator = torch.sum(weight * squared_error)

    if prediction.ndim == 3:
        represented_time_points = (
            prediction.shape[1]
            if weight.shape[1] == 1
            else 1
        )
        denominator = (
            torch.sum(weight)
            * represented_time_points
            * prediction.shape[2]
        )
    else:
        denominator = (
            torch.sum(weight)
            * prediction.shape[1]
        )

    denominator = torch.clamp(
        denominator,
        min=torch.finfo(prediction.dtype).tiny,
    )
    return numerator / denominator


def regression_regularization_penalty(
    model: nn.Module,
    cfg=CFG,
) -> torch.Tensor:
    """
    Return the explicit Ridge/Lasso penalty.

    Bias is not regularized. LSTM, MLP and ordinary Linear return zero
    explicit penalty here. MLP/LSTM optimizer weight decay is configured
    separately in 10_6_trainer.py.
    """
    model_type = str(cfg.MODEL_TYPE).lower().strip()

    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        raise ValueError("Model has no parameters")

    zero = first_parameter.new_zeros(())

    if model_type in ("lstm_fc", "mlp", "linear"):
        return zero

    # Do not use isinstance(model, PointwiseLinearModel) here.  Step-10
    # modules are loaded dynamically by several entry points, so the same
    # source file can legitimately produce distinct Python class objects.
    # In that situation an otherwise valid PointwiseLinearModel would fail an
    # isinstance check merely because it came from another module instance.
    # Validate the required regression interface instead.
    declared_model_type = str(
        getattr(model, "model_type", "")
    ).lower().strip()
    if declared_model_type != model_type:
        raise TypeError(
            "Config MODEL_TYPE=%r, but model declares model_type=%r"
            % (model_type, declared_model_type)
        )

    if not hasattr(model, "regression_weight"):
        raise TypeError(
            "%s regularization requires a model exposing "
            "the regression_weight property, got %s"
            % (model_type, type(model).__name__)
        )

    weight = model.regression_weight
    if not torch.is_tensor(weight):
        raise TypeError("model.regression_weight must be a torch.Tensor")
    if weight.ndim != 2:
        raise ValueError(
            "model.regression_weight must be rank 2, got shape %s"
            % (tuple(weight.shape),)
        )
    if not torch.is_floating_point(weight):
        raise TypeError("model.regression_weight must be floating point")

    if model_type == "ridge":
        alpha = float(cfg.RIDGE_ALPHA)
        if alpha < 0.0:
            raise ValueError("RIDGE_ALPHA cannot be negative")
        return weight.pow(2).sum() * alpha

    if model_type == "lasso":
        alpha = float(cfg.LASSO_ALPHA)
        if alpha < 0.0:
            raise ValueError("LASSO_ALPHA cannot be negative")
        return weight.abs().sum() * alpha

    raise ValueError(
        "Unsupported MODEL_TYPE=%r" % model_type
    )


@dataclass(frozen=True)
class LossBreakdown:
    total_loss: torch.Tensor
    data_loss: torch.Tensor
    regularization_loss: torch.Tensor


def compute_training_loss(
    prediction_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    sample_weight: torch.Tensor,
    model: nn.Module,
    cfg=CFG,
) -> LossBreakdown:
    """
    Compute the data-only objective plus optional classical regularization.

    Physical metrics must not be calculated from this normalized-space loss.
    """
    data_loss = weighted_mse_loss(
        prediction=prediction_normalized,
        target=target_normalized,
        sample_weight=sample_weight,
    )
    regularization = regression_regularization_penalty(
        model=model,
        cfg=cfg,
    )
    total = data_loss + regularization

    return LossBreakdown(
        total_loss=total,
        data_loss=data_loss,
        regularization_loss=regularization,
    )


# =============================================================================
# 3. Streaming physical-unit metric accumulator
# =============================================================================

class StressMetricAccumulator:
    """
    Streaming accumulator for the exact Step-09-compatible metric.

    update() accepts physical-unit stresses. No predictions are retained.
    """

    def __init__(
        self,
        exclude_first_step: bool,
        denominator_floor: float = 1.0e-30,
    ) -> None:
        if denominator_floor <= 0.0:
            raise ValueError("denominator_floor must be positive")

        self.exclude_first_step = bool(exclude_first_step)
        self.denominator_floor = float(denominator_floor)
        self.reset()

    def reset(self) -> None:
        self.trajectory_or_point_count = 0
        self.vector_observation_count = 0
        self.scalar_observation_count = 0

        self.weight_sum_vector = 0.0
        self.weight_sum_scalar = 0.0

        self.weighted_sse = 0.0
        self.weighted_target_sq = 0.0
        self.weighted_abs_error = 0.0
        self.weighted_abs_target = 0.0

        self.unweighted_sse = 0.0
        self.unweighted_target_sq = 0.0
        self.unweighted_abs_error = 0.0
        self.unweighted_abs_target = 0.0

        self.weighted_sse_component = np.zeros(
            TARGET_DIM,
            dtype=np.float64,
        )
        self.weighted_target_sq_component = np.zeros(
            TARGET_DIM,
            dtype=np.float64,
        )
        self.unweighted_sse_component = np.zeros(
            TARGET_DIM,
            dtype=np.float64,
        )
        self.unweighted_target_sq_component = np.zeros(
            TARGET_DIM,
            dtype=np.float64,
        )

    @torch.no_grad()
    def update(
        self,
        prediction_physical: torch.Tensor,
        target_physical: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> None:
        _validate_prediction_target(
            prediction_physical,
            target_physical,
        )

        prediction = prediction_physical.detach()
        target = target_physical.detach()

        if prediction.ndim == 3 and self.exclude_first_step:
            if prediction.shape[1] < 2:
                raise ValueError(
                    "Cannot exclude first step from sequence length < 2"
                )
            prediction = prediction[:, 1:, :]
            target = target[:, 1:, :]

        elif prediction.ndim == 2 and self.exclude_first_step:
            raise ValueError(
                "For pointwise data, exclude time index 0 before calling "
                "update() and construct the accumulator with "
                "exclude_first_step=False."
            )

        if torch.any(~torch.isfinite(prediction)):
            raise ValueError("Prediction contains NaN/Inf")
        if torch.any(~torch.isfinite(target)):
            raise ValueError("Target contains NaN/Inf")

        weight = _expanded_weight_tensor(
            sample_weight,
            prediction,
        )
        if torch.any(~torch.isfinite(weight)):
            raise ValueError("Sample weights contain NaN/Inf")
        if torch.any(weight <= 0):
            raise ValueError("Sample weights must be positive")

        error = prediction - target
        squared_error = error.pow(2)
        squared_target = target.pow(2)
        absolute_error = error.abs()
        absolute_target = target.abs()

        self.weighted_sse += float(
            torch.sum(weight * squared_error).item()
        )
        self.weighted_target_sq += float(
            torch.sum(weight * squared_target).item()
        )
        self.weighted_abs_error += float(
            torch.sum(weight * absolute_error).item()
        )
        self.weighted_abs_target += float(
            torch.sum(weight * absolute_target).item()
        )

        self.unweighted_sse += float(
            torch.sum(squared_error).item()
        )
        self.unweighted_target_sq += float(
            torch.sum(squared_target).item()
        )
        self.unweighted_abs_error += float(
            torch.sum(absolute_error).item()
        )
        self.unweighted_abs_target += float(
            torch.sum(absolute_target).item()
        )

        weighted_sse_component = torch.sum(
            weight * squared_error,
            dim=tuple(range(prediction.ndim - 1)),
        )
        weighted_target_component = torch.sum(
            weight * squared_target,
            dim=tuple(range(prediction.ndim - 1)),
        )
        unweighted_sse_component = torch.sum(
            squared_error,
            dim=tuple(range(prediction.ndim - 1)),
        )
        unweighted_target_component = torch.sum(
            squared_target,
            dim=tuple(range(prediction.ndim - 1)),
        )

        self.weighted_sse_component += (
            weighted_sse_component
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        self.weighted_target_sq_component += (
            weighted_target_component
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        self.unweighted_sse_component += (
            unweighted_sse_component
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        self.unweighted_target_sq_component += (
            unweighted_target_component
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )

        if prediction.ndim == 3:
            batch, length, components = prediction.shape
            vector_count = int(batch * length)

            if weight.shape[1] == 1:
                weight_sum_vector = float(
                    torch.sum(weight).item()
                ) * length
            else:
                weight_sum_vector = float(
                    torch.sum(weight).item()
                )

            self.trajectory_or_point_count += int(batch)
        else:
            points, components = prediction.shape
            vector_count = int(points)
            weight_sum_vector = float(
                torch.sum(weight).item()
            )
            self.trajectory_or_point_count += int(points)

        self.vector_observation_count += vector_count
        self.scalar_observation_count += (
            vector_count * components
        )
        self.weight_sum_vector += weight_sum_vector
        self.weight_sum_scalar += (
            weight_sum_vector * components
        )

    def _sqrt_ratio(
        self,
        numerator: float,
        denominator: float,
    ) -> float:
        if denominator <= self.denominator_floor:
            return float("nan")
        ratio = numerator / denominator
        if ratio < 0.0 or not math.isfinite(ratio):
            return float("nan")
        return math.sqrt(ratio)

    def _ratio(
        self,
        numerator: float,
        denominator: float,
    ) -> float:
        if denominator <= self.denominator_floor:
            return float("nan")
        value = numerator / denominator
        if not math.isfinite(value):
            return float("nan")
        return value

    def compute(self) -> Dict[str, object]:
        if self.scalar_observation_count <= 0:
            raise RuntimeError(
                "No metric observations have been accumulated"
            )

        weighted_component = np.full(
            TARGET_DIM,
            np.nan,
            dtype=np.float64,
        )
        unweighted_component = np.full(
            TARGET_DIM,
            np.nan,
            dtype=np.float64,
        )

        valid_weighted = (
            self.weighted_target_sq_component
            > self.denominator_floor
        )
        valid_unweighted = (
            self.unweighted_target_sq_component
            > self.denominator_floor
        )

        weighted_component[valid_weighted] = (
            100.0
            * np.sqrt(
                self.weighted_sse_component[valid_weighted]
                / self.weighted_target_sq_component[valid_weighted]
            )
        )
        unweighted_component[valid_unweighted] = (
            100.0
            * np.sqrt(
                self.unweighted_sse_component[valid_unweighted]
                / self.unweighted_target_sq_component[valid_unweighted]
            )
        )

        return {
            "weighted_global_relative_l2_pct":
                100.0
                * self._sqrt_ratio(
                    self.weighted_sse,
                    self.weighted_target_sq,
                ),
            "weighted_global_relative_l1_pct":
                100.0
                * self._ratio(
                    self.weighted_abs_error,
                    self.weighted_abs_target,
                ),
            "weighted_rmse":
                self._sqrt_ratio(
                    self.weighted_sse,
                    self.weight_sum_scalar,
                ),
            "weighted_mae":
                self._ratio(
                    self.weighted_abs_error,
                    self.weight_sum_scalar,
                ),

            "unweighted_global_relative_l2_pct":
                100.0
                * self._sqrt_ratio(
                    self.unweighted_sse,
                    self.unweighted_target_sq,
                ),
            "unweighted_global_relative_l1_pct":
                100.0
                * self._ratio(
                    self.unweighted_abs_error,
                    self.unweighted_abs_target,
                ),
            "unweighted_rmse":
                self._sqrt_ratio(
                    self.unweighted_sse,
                    float(self.scalar_observation_count),
                ),
            "unweighted_mae":
                self._ratio(
                    self.unweighted_abs_error,
                    float(self.scalar_observation_count),
                ),

            "weighted_component_relative_l2_pct":
                weighted_component,
            "unweighted_component_relative_l2_pct":
                unweighted_component,
            "component_names": TARGET_ORDER,

            "trajectory_or_point_count":
                int(self.trajectory_or_point_count),
            "vector_observation_count":
                int(self.vector_observation_count),
            "scalar_observation_count":
                int(self.scalar_observation_count),
            "first_step_excluded":
                bool(self.exclude_first_step),
        }


# =============================================================================
# 4. Normalized prediction -> physical metric helper
# =============================================================================

@torch.no_grad()
def update_physical_metric_from_normalized(
    accumulator: StressMetricAccumulator,
    prediction_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    sample_weight: torch.Tensor,
    torch_normalizer,
) -> None:
    """
    Denormalize both prediction and target, then update the physical metric.

    torch_normalizer must provide:
        denormalize_target(tensor)
    """
    prediction_physical = (
        torch_normalizer.denormalize_target(
            prediction_normalized
        )
    )
    target_physical = (
        torch_normalizer.denormalize_target(
            target_normalized
        )
    )

    accumulator.update(
        prediction_physical=prediction_physical,
        target_physical=target_physical,
        sample_weight=sample_weight,
    )


# =============================================================================
# 5. LCS crossing and best-metric tracking
# =============================================================================

@dataclass
class LCSCrossingTracker:
    lcs_line_pct: float

    first_below_epoch: Optional[int] = None
    first_below_elapsed_seconds: Optional[float] = None
    first_below_metric_pct: Optional[float] = None

    best_epoch: Optional[int] = None
    best_elapsed_seconds: Optional[float] = None
    best_metric_pct: float = float("inf")

    last_epoch: Optional[int] = None
    last_metric_pct: Optional[float] = None

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.lcs_line_pct))
            or float(self.lcs_line_pct) < 0.0
        ):
            raise ValueError(
                "Invalid LCS line: %r" % self.lcs_line_pct
            )

    def update(
        self,
        epoch: int,
        elapsed_seconds: float,
        validation_metric_pct: float,
    ) -> Dict[str, object]:
        epoch = int(epoch)
        elapsed_seconds = float(elapsed_seconds)
        metric = float(validation_metric_pct)

        if epoch <= 0:
            raise ValueError("epoch must be positive")
        if elapsed_seconds < 0.0:
            raise ValueError(
                "elapsed_seconds cannot be negative"
            )
        if not math.isfinite(metric):
            raise ValueError(
                "validation_metric_pct must be finite"
            )

        self.last_epoch = epoch
        self.last_metric_pct = metric

        is_below = metric < float(self.lcs_line_pct)

        if metric < self.best_metric_pct:
            self.best_metric_pct = metric
            self.best_epoch = epoch
            self.best_elapsed_seconds = elapsed_seconds

        just_crossed = False
        if is_below and self.first_below_epoch is None:
            self.first_below_epoch = epoch
            self.first_below_elapsed_seconds = elapsed_seconds
            self.first_below_metric_pct = metric
            just_crossed = True

        return {
            "is_below_lcs": is_below,
            "just_crossed_lcs": just_crossed,
            "margin_to_lcs_pct_point":
                metric - float(self.lcs_line_pct),
            "first_below_epoch":
                self.first_below_epoch,
            "first_below_elapsed_seconds":
                self.first_below_elapsed_seconds,
            "best_epoch":
                self.best_epoch,
            "best_metric_pct":
                self.best_metric_pct,
        }

    def state_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> "LCSCrossingTracker":
        tracker = cls(
            lcs_line_pct=float(state["lcs_line_pct"])
        )
        tracker.first_below_epoch = (
            None
            if state.get("first_below_epoch") is None
            else int(state["first_below_epoch"])
        )
        tracker.first_below_elapsed_seconds = (
            None
            if state.get("first_below_elapsed_seconds") is None
            else float(state["first_below_elapsed_seconds"])
        )
        tracker.first_below_metric_pct = (
            None
            if state.get("first_below_metric_pct") is None
            else float(state["first_below_metric_pct"])
        )
        tracker.best_epoch = (
            None
            if state.get("best_epoch") is None
            else int(state["best_epoch"])
        )
        tracker.best_elapsed_seconds = (
            None
            if state.get("best_elapsed_seconds") is None
            else float(state["best_elapsed_seconds"])
        )
        tracker.best_metric_pct = float(
            state.get("best_metric_pct", float("inf"))
        )
        tracker.last_epoch = (
            None
            if state.get("last_epoch") is None
            else int(state["last_epoch"])
        )
        tracker.last_metric_pct = (
            None
            if state.get("last_metric_pct") is None
            else float(state["last_metric_pct"])
        )
        return tracker


# =============================================================================
# 6. Epoch record helper
# =============================================================================

def build_epoch_record(
    epoch: int,
    elapsed_seconds: float,
    learning_rate: float,
    train_total_loss: float,
    train_data_loss: float,
    train_regularization_loss: float,
    train_metric_pct: float,
    val_total_loss: float,
    val_data_loss: float,
    val_regularization_loss: float,
    val_metric_pct: float,
    lcs_line_pct: float,
    crossing_update: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "epoch": int(epoch),
        "elapsed_seconds": float(elapsed_seconds),
        "learning_rate": float(learning_rate),

        "train_total_loss": float(train_total_loss),
        "train_data_loss": float(train_data_loss),
        "train_regularization_loss": float(
            train_regularization_loss
        ),
        "train_metric_pct": float(train_metric_pct),

        "val_total_loss": float(val_total_loss),
        "val_data_loss": float(val_data_loss),
        "val_regularization_loss": float(
            val_regularization_loss
        ),
        "val_metric_pct": float(val_metric_pct),

        "lcs_line_pct": float(lcs_line_pct),
        "val_minus_lcs_pct_point":
            float(val_metric_pct) - float(lcs_line_pct),
        "is_below_lcs": bool(
            crossing_update["is_below_lcs"]
        ),
        "just_crossed_lcs": bool(
            crossing_update["just_crossed_lcs"]
        ),
        "first_below_lcs_epoch":
            crossing_update["first_below_epoch"],
        "first_below_lcs_elapsed_seconds":
            crossing_update["first_below_elapsed_seconds"],
        "best_epoch": crossing_update["best_epoch"],
        "best_val_metric_pct":
            crossing_update["best_metric_pct"],
    }


def save_metric_definition(
    output_path: Path | str,
    cfg=CFG,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_name": "CPFE_STEP10_LOSS_AND_METRIC_V1",
        "model_type": str(cfg.MODEL_TYPE),
        "training_data_loss": "sample_weighted_MSE_in_normalized_target_space",
        "physics_loss": False,
        "mlp_explicit_penalty": False,
        "ridge_penalty": "RIDGE_ALPHA * sum(regression_weight^2)",
        "lasso_penalty": "LASSO_ALPHA * sum(abs(regression_weight))",
        "bias_regularized": False,
        "validation_metric":
            "weighted_global_relative_l2_pct_in_physical_stress_units",
        "validation_formula":
            "100*sqrt(sum(w*(sigma_pred-sigma_true)^2)/sum(w*sigma_true^2))",
        "lcs_comparison_first_step_excluded": True,
        "component_order": list(TARGET_ORDER),
    }

    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


# =============================================================================
# 7. Smoke tests
# =============================================================================

def _manual_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    p = prediction.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    w = weights.detach().cpu().numpy()

    if p.ndim == 3:
        numerator = np.sum(
            w[:, None, None] * (p - t) ** 2
        )
        denominator = (
            np.sum(w) * p.shape[1] * p.shape[2]
        )
    else:
        numerator = np.sum(
            w[:, None] * (p - t) ** 2
        )
        denominator = np.sum(w) * p.shape[1]

    return float(numerator / denominator)


def smoke_test_losses_and_metrics(cfg=CFG) -> None:
    torch.manual_seed(123)

    prediction = torch.randn(4, 5, TARGET_DIM)
    target = torch.randn(4, 5, TARGET_DIM)
    weights = torch.tensor(
        [1.0, 2.0, 0.5, 3.0],
        dtype=torch.float32,
    )

    loss = weighted_mse_loss(
        prediction,
        target,
        weights,
    )
    manual = _manual_weighted_mse(
        prediction,
        target,
        weights,
    )
    if not math.isclose(
        float(loss.item()),
        manual,
        rel_tol=1.0e-6,
        abs_tol=1.0e-7,
    ):
        raise RuntimeError(
            "Weighted MSE mismatch: torch=%r manual=%r"
            % (loss.item(), manual)
        )

    accumulator = StressMetricAccumulator(
        exclude_first_step=True
    )
    accumulator.update(
        prediction_physical=prediction,
        target_physical=target,
        sample_weight=weights,
    )
    metrics = accumulator.compute()

    p = prediction[:, 1:, :].numpy()
    t = target[:, 1:, :].numpy()
    w = weights.numpy()[:, None, None]

    manual_relative = (
        100.0
        * math.sqrt(
            float(np.sum(w * (p - t) ** 2))
            / float(np.sum(w * t ** 2))
        )
    )

    if not math.isclose(
        float(metrics["weighted_global_relative_l2_pct"]),
        manual_relative,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    ):
        raise RuntimeError(
            "Relative-L2 mismatch: accumulator=%r manual=%r"
            % (
                metrics["weighted_global_relative_l2_pct"],
                manual_relative,
            )
        )

    tracker = LCSCrossingTracker(
        lcs_line_pct=10.0
    )
    state1 = tracker.update(
        epoch=1,
        elapsed_seconds=2.0,
        validation_metric_pct=12.0,
    )
    state2 = tracker.update(
        epoch=2,
        elapsed_seconds=4.0,
        validation_metric_pct=9.0,
    )
    state3 = tracker.update(
        epoch=3,
        elapsed_seconds=6.0,
        validation_metric_pct=8.0,
    )

    if state1["is_below_lcs"]:
        raise RuntimeError("Unexpected LCS crossing at epoch 1")
    if not state2["just_crossed_lcs"]:
        raise RuntimeError("Expected first LCS crossing at epoch 2")
    if state3["just_crossed_lcs"]:
        raise RuntimeError(
            "LCS crossing must only be recorded once"
        )
    if tracker.first_below_epoch != 2:
        raise RuntimeError("Incorrect first_below_epoch")
    if tracker.best_epoch != 3:
        raise RuntimeError("Incorrect best_epoch")

    # Cross-module compatibility smoke test.  This intentionally uses a
    # minimal model-like object rather than this module's own imported class,
    # reproducing the dynamic-import situation used by 10_6_trainer.py.
    class ForeignPointwiseModel(nn.Module):
        model_type = "ridge"

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 2, bias=True)

        @property
        def regression_weight(self) -> torch.Tensor:
            return self.linear.weight

    foreign_model = ForeignPointwiseModel()

    class RidgeSmokeConfig:
        MODEL_TYPE = "ridge"
        RIDGE_ALPHA = 0.25
        LASSO_ALPHA = 0.1

    ridge_penalty = regression_regularization_penalty(
        foreign_model,
        RidgeSmokeConfig,
    )
    ridge_expected = (
        foreign_model.regression_weight.pow(2).sum() * 0.25
    )
    if not torch.allclose(ridge_penalty, ridge_expected):
        raise RuntimeError("Cross-module Ridge penalty test failed")

    foreign_model.model_type = "lasso"

    class LassoSmokeConfig:
        MODEL_TYPE = "lasso"
        RIDGE_ALPHA = 0.25
        LASSO_ALPHA = 0.1

    lasso_penalty = regression_regularization_penalty(
        foreign_model,
        LassoSmokeConfig,
    )
    lasso_expected = (
        foreign_model.regression_weight.abs().sum() * 0.1
    )
    if not torch.allclose(lasso_penalty, lasso_expected):
        raise RuntimeError("Cross-module Lasso penalty test failed")

    print("Loss/metric smoke test: PASS")


# =============================================================================
# 8. Standalone entry
# =============================================================================

if __name__ == "__main__":
    CFG.print_summary()
    smoke_test_losses_and_metrics(CFG)

    CFG.create_output_directories()
    definition_path = save_metric_definition(
        Path(CFG.EXPERIMENT_ROOT)
        / "loss_and_metric_definition.json",
        CFG,
    )
    print("Definition saved:", definition_path)
