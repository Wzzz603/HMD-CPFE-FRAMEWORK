#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_4_models.py

Unified Step-10 model definitions.

Models
------
1. lstm_fc
   Full sequence model:
       [B, L, 22]
       -> two stacked LSTM layers, hidden size 256
       -> one pointwise Linear(256, 6) output layer
       -> [B, L, 6]

2. linear
3. ridge
4. lasso
   Current-step-only baselines:
       [B, L, 22]
       -> the same Linear(22, 6) is applied independently at every time step
       -> [B, L, 6]

Linear, Ridge and Lasso intentionally share the same affine model container.
Their fitted coefficients are produced by the classical one-shot baseline
solver in 10_6_trainer.py:

    linear : streamed weighted ordinary least squares
    ridge  : streamed weighted ridge solve
    lasso  : streamed Gram matrix plus coordinate descent

The shared PyTorch container is used only for common prediction, checkpoint and
export interfaces; the formal baselines are not trained with Adam/AdamW.

Important
---------
- No physics-informed layer or physics loss is used.
- BCC and HCP are trained separately.
- A multiplier-k model only reads multiplier-k data.
- Sample_Weight is not a model input.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn


# =============================================================================
# 0. Dynamic import of 10_1_config.py
# =============================================================================

_THIS_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _THIS_DIR / "10_1_config.py"

if not _CONFIG_PATH.is_file():
    raise FileNotFoundError(
        "10_1_config.py must be in the same directory as this file: %s"
        % _CONFIG_PATH
    )

_spec = importlib.util.spec_from_file_location(
    "step10_config_for_models",
    _CONFIG_PATH,
)
if _spec is None or _spec.loader is None:
    raise ImportError(
        "Could not create import specification for %s" % _CONFIG_PATH
    )

_config_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _config_module
_spec.loader.exec_module(_config_module)

CFG = _config_module.CFG
INPUT_DIM = int(_config_module.INPUT_DIM)
TARGET_DIM = int(_config_module.TARGET_DIM)
MODEL_DISPLAY_NAMES = dict(_config_module.MODEL_DISPLAY_NAMES)


# =============================================================================
# 1. Model information
# =============================================================================

@dataclass(frozen=True)
class ModelInfo:
    model_type: str
    display_name: str
    input_dim: int
    target_dim: int
    parameter_count_total: int
    parameter_count_trainable: int
    uses_sequence_history: bool
    recurrent_layers: int
    bidirectional: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_type": self.model_type,
            "display_name": self.display_name,
            "input_dim": self.input_dim,
            "target_dim": self.target_dim,
            "parameter_count_total": self.parameter_count_total,
            "parameter_count_trainable": self.parameter_count_trainable,
            "uses_sequence_history": self.uses_sequence_history,
            "recurrent_layers": self.recurrent_layers,
            "bidirectional": self.bidirectional,
        }


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(
        int(p.numel())
        for p in model.parameters()
        if p.requires_grad
    )
    return total, trainable


# =============================================================================
# 2. Initialization helpers
# =============================================================================


def initialize_linear_layer(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def initialize_lstm(lstm: nn.LSTM) -> None:
    """
    Stable initialization:
    - input weights: Xavier uniform;
    - recurrent weights: orthogonal;
    - biases: zero;
    - forget-gate bias: one.
    """
    for name, parameter in lstm.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(parameter)
        elif "weight_hh" in name:
            nn.init.orthogonal_(parameter)
        elif "bias" in name:
            nn.init.zeros_(parameter)

            # PyTorch gate order:
            # input, forget, cell, output
            hidden = parameter.numel() // 4
            with torch.no_grad():
                parameter[hidden:2 * hidden].fill_(1.0)


# =============================================================================
# 3. LSTM + FC
# =============================================================================

class LSTMFCModel(nn.Module):
    """
    Sequence-aware stress predictor.

    Input
    -----
    x: [B, L, 22]

    Output
    ------
    sigma_pred: [B, L, 6]
    """

    model_type = "lstm_fc"
    uses_sequence_history = True

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        hidden_size: int,
        num_layers: int,
        lstm_dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if target_dim <= 0:
            raise ValueError("target_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= lstm_dropout < 1.0:
            raise ValueError("lstm_dropout must be in [0, 1)")
        if num_layers == 1 and lstm_dropout != 0.0:
            raise ValueError(
                "Set lstm_dropout=0.0 when num_layers=1"
            )

        self.input_dim = int(input_dim)
        self.target_dim = int(target_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1

        recurrent_dropout = (
            float(lstm_dropout)
            if self.num_layers > 1
            else 0.0
        )

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=self.bidirectional,
        )

        lstm_output_dim = (
            self.hidden_size * self.num_directions
        )

        # Exactly one fully connected output layer:
        #     [B,L,lstm_output_dim] -> [B,L,target_dim]
        #
        # nn.Linear acts on the final tensor dimension, so the same 256 -> 6
        # mapping is applied independently to every time point.
        self.fc_head = nn.Linear(
            lstm_output_dim,
            self.target_dim,
        )

        initialize_lstm(self.lstm)
        initialize_linear_layer(self.fc_head)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        return_hidden: bool = False,
    ):
        if x.ndim != 3:
            raise ValueError(
                "LSTM input must be [B,L,%d], got %s"
                % (self.input_dim, tuple(x.shape))
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                "LSTM input dimension=%d, expected=%d"
                % (x.shape[-1], self.input_dim)
            )

        recurrent_output, final_hidden = self.lstm(
            x,
            hidden_state,
        )
        sigma_pred = self.fc_head(recurrent_output)

        if sigma_pred.shape[-1] != self.target_dim:
            raise RuntimeError(
                "Output dimension=%d, expected=%d"
                % (sigma_pred.shape[-1], self.target_dim)
            )

        if return_hidden:
            return sigma_pred, final_hidden
        return sigma_pred

    def create_initial_hidden(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = (
            self.num_layers * self.num_directions,
            int(batch_size),
            self.hidden_size,
        )
        h0 = torch.zeros(
            shape,
            dtype=dtype,
            device=device,
        )
        c0 = torch.zeros(
            shape,
            dtype=dtype,
            device=device,
        )
        return h0, c0


# =============================================================================
# 4. MLP current-step-only nonlinear model
# =============================================================================

class MLPModel(nn.Module):
    """
    Pointwise nonlinear stress predictor.

    Architecture
    ------------
        22 -> 256 -> 256 -> 6

    Accepted input
    --------------
        [P, 22] or [B, L, 22]

    For [B,L,22], all Linear/GELU layers act only on the last dimension.
    Therefore prediction at time t depends only on x[t]. There is no recurrent
    state, temporal pooling, convolution, or attention across time.
    """

    model_type = "mlp"
    uses_sequence_history = False

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        hidden_size_1: int,
        hidden_size_2: int,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if input_dim <= 0 or target_dim <= 0:
            raise ValueError("input_dim and target_dim must be positive")
        if hidden_size_1 <= 0 or hidden_size_2 <= 0:
            raise ValueError("MLP hidden sizes must be positive")
        if str(activation).lower().strip() != "gelu":
            raise ValueError("Only activation='gelu' is currently supported")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")

        self.input_dim = int(input_dim)
        self.target_dim = int(target_dim)
        self.hidden_size_1 = int(hidden_size_1)
        self.hidden_size_2 = int(hidden_size_2)
        self.dropout_probability = float(dropout)

        self.fc1 = nn.Linear(self.input_dim, self.hidden_size_1)
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc_out = nn.Linear(self.hidden_size_2, self.target_dim)

        self.activation = nn.GELU()
        self.dropout = (
            nn.Dropout(self.dropout_probability)
            if self.dropout_probability > 0.0
            else nn.Identity()
        )

        initialize_linear_layer(self.fc1)
        initialize_linear_layer(self.fc2)
        initialize_linear_layer(self.fc_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (2, 3):
            raise ValueError(
                "MLP input must be [P,%d] or [B,L,%d], got %s"
                % (self.input_dim, self.input_dim, tuple(x.shape))
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                "MLP input dimension=%d, expected=%d"
                % (x.shape[-1], self.input_dim)
            )

        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        output = self.fc_out(x)

        if output.shape[-1] != self.target_dim:
            raise RuntimeError(
                "MLP output dimension=%d, expected=%d"
                % (output.shape[-1], self.target_dim)
            )

        return output


# =============================================================================
# 5. Linear / Ridge / Lasso pointwise model
# =============================================================================

class PointwiseLinearModel(nn.Module):
    """
    Current-step-only baseline.

    The same affine mapping is independently applied to every time point:

        sigma_hat[t] = W x[t] + b

    There is no hidden state and no sequence history.
    """

    uses_sequence_history = False

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        model_type: str,
        fit_intercept: bool = True,
    ) -> None:
        super().__init__()

        model_type = str(model_type).lower().strip()
        if model_type not in ("linear", "ridge", "lasso"):
            raise ValueError(
                "PointwiseLinearModel requires "
                "linear/ridge/lasso, got %r" % model_type
            )

        if input_dim <= 0 or target_dim <= 0:
            raise ValueError(
                "input_dim and target_dim must be positive"
            )

        self.model_type = model_type
        self.input_dim = int(input_dim)
        self.target_dim = int(target_dim)

        self.linear = nn.Linear(
            self.input_dim,
            self.target_dim,
            bias=bool(fit_intercept),
        )
        initialize_linear_layer(self.linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (2, 3):
            raise ValueError(
                "Pointwise model input must be [P,%d] or "
                "[B,L,%d], got %s"
                % (
                    self.input_dim,
                    self.input_dim,
                    tuple(x.shape),
                )
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                "Pointwise input dimension=%d, expected=%d"
                % (x.shape[-1], self.input_dim)
            )

        output = self.linear(x)

        if output.shape[-1] != self.target_dim:
            raise RuntimeError(
                "Pointwise output dimension=%d, expected=%d"
                % (output.shape[-1], self.target_dim)
            )

        return output

    @property
    def regression_weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def regression_bias(self) -> Optional[torch.Tensor]:
        return self.linear.bias


# =============================================================================
# 6. Model factory
# =============================================================================

def build_model(cfg=CFG) -> nn.Module:
    """
    Build exactly one model selected in 10_1_config.py.
    """
    model_type = str(cfg.MODEL_TYPE).lower().strip()

    if model_type == "lstm_fc":
        model = LSTMFCModel(
            input_dim=INPUT_DIM,
            target_dim=TARGET_DIM,
            hidden_size=int(cfg.LSTM_HIDDEN_SIZE),
            num_layers=int(cfg.LSTM_NUM_LAYERS),
            lstm_dropout=float(cfg.LSTM_DROPOUT),
            bidirectional=bool(cfg.LSTM_BIDIRECTIONAL),
        )

    elif model_type == "mlp":
        model = MLPModel(
            input_dim=INPUT_DIM,
            target_dim=TARGET_DIM,
            hidden_size_1=int(cfg.MLP_HIDDEN_SIZE_1),
            hidden_size_2=int(cfg.MLP_HIDDEN_SIZE_2),
            activation=str(cfg.MLP_ACTIVATION),
            dropout=float(cfg.MLP_DROPOUT),
        )

    elif model_type == "linear":
        model = PointwiseLinearModel(
            input_dim=INPUT_DIM,
            target_dim=TARGET_DIM,
            model_type="linear",
            fit_intercept=bool(cfg.LINEAR_FIT_INTERCEPT),
        )

    elif model_type == "ridge":
        model = PointwiseLinearModel(
            input_dim=INPUT_DIM,
            target_dim=TARGET_DIM,
            model_type="ridge",
            fit_intercept=bool(cfg.RIDGE_FIT_INTERCEPT),
        )

    elif model_type == "lasso":
        model = PointwiseLinearModel(
            input_dim=INPUT_DIM,
            target_dim=TARGET_DIM,
            model_type="lasso",
            fit_intercept=bool(cfg.LASSO_FIT_INTERCEPT),
        )

    else:
        raise ValueError(
            "Unsupported MODEL_TYPE=%r" % model_type
        )

    return model.to(torch.device(cfg.DEVICE))


def get_model_info(
    model: nn.Module,
    cfg=CFG,
) -> ModelInfo:
    total, trainable = count_parameters(model)

    return ModelInfo(
        model_type=str(cfg.MODEL_TYPE),
        display_name=MODEL_DISPLAY_NAMES[
            str(cfg.MODEL_TYPE)
        ],
        input_dim=INPUT_DIM,
        target_dim=TARGET_DIM,
        parameter_count_total=total,
        parameter_count_trainable=trainable,
        uses_sequence_history=bool(
            getattr(model, "uses_sequence_history", False)
        ),
        recurrent_layers=(
            int(cfg.LSTM_NUM_LAYERS)
            if str(cfg.MODEL_TYPE) == "lstm_fc"
            else 0
        ),
        bidirectional=(
            bool(cfg.LSTM_BIDIRECTIONAL)
            if str(cfg.MODEL_TYPE) == "lstm_fc"
            else False
        ),
    )


def save_model_structure(
    model: nn.Module,
    cfg=CFG,
) -> Path:
    cfg.create_output_directories()

    info = get_model_info(model, cfg)
    output_path = (
        Path(cfg.EXPERIMENT_ROOT)
        / "model_structure.json"
    )

    payload = {
        **info.to_dict(),
        "model_repr": str(model),
        "architecture_tag": str(cfg.MODEL_ARCHITECTURE_TAG),
        "output_head": (
            "two_hidden_layer_mlp_plus_linear_output"
            if str(cfg.MODEL_TYPE) == "mlp"
            else "single_linear_output_layer"
        ),
        "phase": str(cfg.PHASE),
        "multiplier": int(cfg.MULTIPLIER),
        "sequence_length": int(
            cfg.expected_sequence_length
        ),
        "input_order": list(
            _config_module.INPUT_ORDER
        ),
        "target_order": list(
            _config_module.TARGET_ORDER
        ),
        "physics_informed": False,
        "output_mode": (
            str(cfg.LSTM_OUTPUT_MODE)
            if str(cfg.MODEL_TYPE) == "lstm_fc"
            else (
                str(cfg.MLP_OUTPUT_MODE)
                if str(cfg.MODEL_TYPE) == "mlp"
                else "absolute_stress"
            )
        ),
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

def smoke_test_model(
    model: nn.Module,
    cfg=CFG,
) -> None:
    device = torch.device(cfg.DEVICE)

    batch_size = 3
    sequence_length = int(
        cfg.expected_sequence_length
    )

    x_sequence = torch.randn(
        batch_size,
        sequence_length,
        INPUT_DIM,
        dtype=torch.float32,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        y_sequence = model(x_sequence)

    expected_sequence_shape = (
        batch_size,
        sequence_length,
        TARGET_DIM,
    )
    if tuple(y_sequence.shape) != expected_sequence_shape:
        raise RuntimeError(
            "Sequence output shape=%s, expected=%s"
            % (
                tuple(y_sequence.shape),
                expected_sequence_shape,
            )
        )
    if not torch.isfinite(y_sequence).all():
        raise RuntimeError(
            "NaN/Inf found in model sequence output"
        )

    if str(cfg.MODEL_TYPE) in (
        "mlp",
        "linear",
        "ridge",
        "lasso",
    ):
        x_points = torch.randn(
            11,
            INPUT_DIM,
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            y_points = model(x_points)

        if tuple(y_points.shape) != (11, TARGET_DIM):
            raise RuntimeError(
                "Pointwise output shape=%s, expected=%s"
                % (
                    tuple(y_points.shape),
                    (11, TARGET_DIM),
                )
            )

        # Verify current-step independence: changing one time point must not
        # alter predictions at other time points.
        x_a = torch.randn(
            2,
            7,
            INPUT_DIM,
            device=device,
        )
        x_b = x_a.clone()
        x_b[:, 3, :] += 1.0

        with torch.no_grad():
            y_a = model(x_a)
            y_b = model(x_b)

        unaffected = torch.ones(
            7,
            dtype=torch.bool,
            device=device,
        )
        unaffected[3] = False

        if not torch.equal(
            y_a[:, unaffected, :],
            y_b[:, unaffected, :],
        ):
            raise RuntimeError(
                "Pointwise model unexpectedly uses sequence history"
            )

    print("Model smoke test: PASS")


# =============================================================================
# 8. Standalone entry
# =============================================================================

if __name__ == "__main__":
    CFG.print_summary()

    model = build_model(CFG)
    info = get_model_info(model, CFG)

    print(model)
    print("-" * 108)
    print("Model type          :", info.model_type)
    print("Display name        :", info.display_name)
    print("Uses history        :", info.uses_sequence_history)
    print("Total parameters    :", info.parameter_count_total)
    print("Trainable parameters:", info.parameter_count_trainable)
    print("Device              :", next(model.parameters()).device)

    structure_path = save_model_structure(
        model,
        CFG,
    )
    smoke_test_model(
        model,
        CFG,
    )

    print("Structure saved     :", structure_path)
