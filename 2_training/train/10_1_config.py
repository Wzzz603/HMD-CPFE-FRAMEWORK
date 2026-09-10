#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_1_config.py

Unified configuration for the Step-10 training series.

Supported models
----------------
1. lstm_fc
   Full sequence architecture:
       [batch, sequence_length, 22]
       -> two stacked LSTM layers, hidden size 256
       -> one Linear(256, 6) output layer
       -> [batch, sequence_length, 6]

2. linear
3. ridge
4. lasso
   Baseline models use current-step features only:
       [sample, 22] -> [sample, 6]
   They do not use sequence history.

Important
---------
- BCC and HCP are trained separately.
- A multiplier-k model reads only multiplier-k data.
- Train/val indices created by Step-08 are reused directly.
- No physics-informed loss is used.
- The LCS horizontal reference line is read from Step-09.
- Validation comparison against LCS uses the same weighted global relative-L2
  metric and excludes the first time point.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


# =============================================================================
# 0. Allowed options
# =============================================================================

ALLOWED_PHASES = ("BCC", "HCP")

ALLOWED_MODEL_TYPES = (
    "lstm_fc",
    "mlp",
    "linear",
    "ridge",
    "lasso",
)

MODEL_DISPLAY_NAMES = {
    "lstm_fc": "2-layer LSTM + single FC output",
    "mlp": "MLP 22-256-256-6",
    "linear": "Linear Regression",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
}

# Fixed Step-08 feature definition.
DYNAMIC_F_DIM = 9
STATIC_INPUT_DIM = 4
THETA_DIM = 9
INPUT_DIM = 22
TARGET_DIM = 6

INPUT_ORDER = (
    "F11", "F12", "F13",
    "F21", "F22", "F23",
    "F31", "F32", "F33",
    "DTIME",
    "Temperature",
    "N_total",
    "N_diff",
    "theta11", "theta12", "theta13",
    "theta21", "theta22", "theta23",
    "theta31", "theta32", "theta33",
)

TARGET_ORDER = (
    "sigma11",
    "sigma22",
    "sigma33",
    "sigma12",
    "sigma13",
    "sigma23",
)

STATIC_COLUMNS_IN_H5 = (
    "DTIME",
    "Temperature",
    "N_total",
    "N_diff",
    "Sample_Weight",
)

SPLIT_CODE = {
    "train": 0,
    "val": 1,
}


# =============================================================================
# 1. User configuration
# =============================================================================

@dataclass
class Config:
    # -------------------------------------------------------------------------
    # Main experiment identity
    # -------------------------------------------------------------------------
    PHASE: str = "HCP"
    MULTIPLIER: int = 1

    # Step-08 base standard-interval count for the current external dataset.
    # Current Step-07/08 logic: 27 base intervals; multiplier=3 -> sequence length=9.
    BASE_SEQUENCE_LENGTH: int = 10

    # Choose one:
    #   lstm_fc / linear / ridge / lasso
    MODEL_TYPE: str = "lstm_fc"

    # Used only as a folder label. It should match the Step-08 run folder.
    # DATASET_RUN_NAME: str = "05_simple_external27""triaxial_np_32X"
    DATASET_RUN_NAME: str = "triaxial_np_32X"
    # -------------------------------------------------------------------------
    # Project paths
    # -------------------------------------------------------------------------
    PROJECT_ROOT: str = (
    r"F:\1_job\20251124\CPFE_PARAMATER_TEST_PBC\2_RVE_BUILD"
    r"\13_triaxial_nonproportional_calc\work\32X"
    )

    # STEP08_ROOT_NAME: str = "08_multiscale_h5_training_dataset"
    # STEP09_ROOT_NAME: str = "09_lcs_baseline"
    # STEP10_ROOT_NAME: str = "10_training"
    STEP08_ROOT_NAME: str = "14_3_h5_training_dataset"
    STEP09_ROOT_NAME: str = "14_4_lcs_baseline"
    STEP10_ROOT_NAME: str = "14_5_training"

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    RANDOM_SEED: int = 20260713
    DETERMINISTIC_TORCH: bool = False

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------
    # "ram":
    #   Load the selected phase H5 into system memory once.
    #
    # "h5":
    #   Stream batches directly from H5.
    #
    # For the current HCP 3x dataset, "ram" is recommended.
    DATA_ACCESS_MODE: str = "ram"

    # DataLoader settings for LSTM.
    BATCH_SIZE: int = 2048
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = False
    DROP_LAST_TRAIN_BATCH: bool = False
    
    # Limit for debugging. None means all selected trajectories.
    MAX_TRAIN_TRAJECTORIES: int | None = None
    MAX_VAL_TRAJECTORIES: int | None = None

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------
    # Statistics must be calculated from train only.
    NORMALIZE_INPUTS: bool = True
    NORMALIZE_TARGETS: bool = True

    # Supported:
    #   standard: (x - mean) / std
    NORMALIZATION_MODE: str = "standard"

    NORMALIZATION_EPS: float = 1.0e-8

    # Streamed trajectory count used when compiling train-only statistics.
    # This controls temporary memory only; it does not subsample the dataset.
    NORMALIZATION_CHUNK_TRAJECTORIES: int = 2048

    # Reuse an existing compatible normalization file by default. Set True only
    # when the Step-08 H5 content has changed and statistics must be rebuilt.
    NORMALIZATION_OVERWRITE: bool = False

    # Mean/std are deliberately unweighted. Sample_Weight remains a loss and
    # evaluation weight, not a normalization-statistics weight.
    NORMALIZATION_USE_SAMPLE_WEIGHT: bool = False

    # Sample_Weight is a loss weight only. It is never part of the 22 inputs.
    USE_SAMPLE_WEIGHT: bool = True

    # -------------------------------------------------------------------------
    # Common validation metric
    # -------------------------------------------------------------------------
    # This must match Step-09 PRIMARY_METRIC.
    VALIDATION_METRIC: str = "weighted_global_relative_l2_pct"

    # Strict LCS comparison excludes the first time point because no previous
    # converged solution exists inside the trajectory.
    EXCLUDE_FIRST_STEP_FOR_LCS_COMPARISON: bool = True

    # -------------------------------------------------------------------------
    # LSTM model: two stacked LSTM layers + one FC output layer
    # -------------------------------------------------------------------------
    # Tensor flow:
    #   [B,L,22]
    #       -> stacked LSTM (2 layers, hidden size 256)
    #       -> [B,L,256]
    #       -> one shared Linear(256,6) applied at every time point
    #       -> [B,L,6]
    LSTM_HIDDEN_SIZE: int = 256
    LSTM_NUM_LAYERS: int = 2
    LSTM_DROPOUT: float = 0
    LSTM_BIDIRECTIONAL: bool = False

    # There is no FC hidden layer, FC activation, or FC dropout.
    # The only FC layer is the final recurrent-output -> 6 stress mapping.
    LSTM_OUTPUT_MODE: str = "absolute_stress"

    # -------------------------------------------------------------------------
    # MLP model: current-step-only nonlinear baseline
    # -------------------------------------------------------------------------
    # Tensor flow:
    #   [B,L,22]
    #       -> Linear(22,256) -> GELU
    #       -> Linear(256,256) -> GELU
    #       -> Linear(256,6)
    #       -> [B,L,6]
    #
    # nn.Linear operates on the final tensor dimension, so every time point is
    # processed independently. No hidden/recurrent state is transferred across
    # time. This makes MLP a nonlinear current-state baseline against LSTM.
    MLP_HIDDEN_SIZE_1: int = 256
    MLP_HIDDEN_SIZE_2: int = 256
    MLP_ACTIVATION: str = "gelu"
    MLP_DROPOUT: float = 0.0
    MLP_OUTPUT_MODE: str = "absolute_stress"

    # Keep neural-network regularization explicit and independent from the
    # classical Ridge/Lasso penalties. Default matches the LSTM weight decay.
    MLP_WEIGHT_DECAY: float = 1.0e-5

    # -------------------------------------------------------------------------
    # Neural-network training (LSTM / MLP)
    # -------------------------------------------------------------------------
    MAX_EPOCHS: int = 2000
    LEARNING_RATE: float = 1.0e-2
    WEIGHT_DECAY: float = 1.0e-5

    OPTIMIZER: str = "adamw"
    LR_SCHEDULER: str = "reduce_on_plateau"

    LR_REDUCE_FACTOR: float = 0.8
    LR_REDUCE_PATIENCE: int = 30
    MIN_LEARNING_RATE: float = 1.0e-6

    EARLY_STOPPING_PATIENCE: int = 45
    EARLY_STOPPING_MIN_DELTA: float = 0.0

    GRADIENT_CLIP_NORM: float = 1.0
    USE_AMP: bool = True

    # Training objective. No physics term.  
    DATA_LOSS: str = "weighted_mse"

    # -------------------------------------------------------------------------
    # Linear / Ridge / Lasso baseline settings
    # -------------------------------------------------------------------------
    # Baselines use current-step information only and do not use LSTM history.
    BASELINE_INPUT_MODE: str = "current_step_only"

    # Baselines normally use all time points as independent regression samples.
    # The first point is kept for fitting, but the reported LCS-comparison metric
    # still excludes it.
    BASELINE_INCLUDE_FIRST_STEP_IN_FIT: bool = True

    # Processing chunk size used when flattening H5 sequences into point samples.
    BASELINE_POINT_CHUNK_SIZE: int = 1_000_000

    # Linear Regression
    LINEAR_FIT_INTERCEPT: bool = True

    # Ridge
    RIDGE_ALPHA: float = 1.0
    RIDGE_FIT_INTERCEPT: bool = True
    RIDGE_SOLVER: str = "auto"

    # Lasso
    # Multi-output Lasso is fitted as one independent model per stress component.
    LASSO_ALPHA: float = 1.0e-4
    LASSO_FIT_INTERCEPT: bool = True
    LASSO_MAX_ITER: int = 5000
    LASSO_TOL: float = 1.0e-4
    LASSO_SELECTION: str = "cyclic"

    # Optional baseline subsampling. Keep None for the formal comparison.
    BASELINE_MAX_TRAIN_POINTS: int | None = None
    BASELINE_MAX_VAL_POINTS: int | None = None

    # -------------------------------------------------------------------------
    # Logging, checkpoints and live dashboard
    # -------------------------------------------------------------------------
    SAVE_BEST_CHECKPOINT: bool = True
    SAVE_LAST_CHECKPOINT: bool = True
    SAVE_EVERY_N_EPOCHS: int = 0

    # Resume policy:
    #   auto     -> resume if the selected checkpoint exists;
    #   never    -> require a fresh experiment directory;
    #   required -> fail unless a checkpoint exists.
    RESUME_MODE: str = "auto"
    RESUME_CHECKPOINT_PATH: str = ""
    STRICT_RESUME_CONFIG: bool = True
    SAVE_INTERRUPTION_NOTE: bool = True

    # The live dashboard will display:
    #   train metric
    #   val metric
    #   fixed LCS val reference line
    ENABLE_LIVE_DASHBOARD: bool = False
    DASHBOARD_SHOW_WINDOW: bool = True
    DASHBOARD_Y_SCALE: str = "linear"
    DASHBOARD_DPI: int = 160
    DASHBOARD_REFRESH_EVERY_EPOCHS: int = 1
    SAVE_DASHBOARD_EVERY_EPOCHS: int = 1

    # Console live table settings. Clearing occurs only for an interactive TTY.
    CLEAR_TERMINAL_FOR_LIVE_TABLE: bool = True
    LIVE_TABLE_ROWS: int = 12

    PRINT_EVERY_EPOCHS: int = 1
    EVALUATE_EVERY_EPOCHS: int = 1

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Strict checks
    # -------------------------------------------------------------------------
    REQUIRE_H5_SPLIT_INDICES: bool = True
    REQUIRE_LCS_REFERENCE: bool = True
    REQUIRE_SEQUENCE_LENGTH_MATCH: bool = True
    FAIL_ON_NAN_OR_INF: bool = True
    FAIL_ON_NONPOSITIVE_WEIGHT: bool = True

    # -------------------------------------------------------------------------
    # Derived paths. Do not edit these directly.
    # -------------------------------------------------------------------------
    H5_DATA_PATH: str = field(init=False)
    LCS_BASELINE_H5_PATH: str = field(init=False)
    EXPERIMENT_ROOT: str = field(init=False)
    CHECKPOINT_DIR: str = field(init=False)
    LOG_DIR: str = field(init=False)
    FIGURE_DIR: str = field(init=False)
    EXPORT_DIR: str = field(init=False)
    NORMALIZATION_PATH: str = field(init=False)
    HISTORY_CSV_PATH: str = field(init=False)
    CONFIG_SNAPSHOT_PATH: str = field(init=False)
    LAST_CHECKPOINT_PATH: str = field(init=False)
    BEST_CHECKPOINT_PATH: str = field(init=False)
    LIVE_DASHBOARD_PATH: str = field(init=False)
    TRAINING_SUMMARY_PATH: str = field(init=False)
    MODEL_ARCHITECTURE_TAG: str = field(init=False)

    def __post_init__(self) -> None:
        self.PHASE = str(self.PHASE).upper().strip()
        self.MODEL_TYPE = str(self.MODEL_TYPE).lower().strip()

        if self.MODEL_TYPE == "lstm_fc":
            direction_tag = (
                "Bi"
                if bool(self.LSTM_BIDIRECTIONAL)
                else "Uni"
            )
            self.MODEL_ARCHITECTURE_TAG = (
                f"{direction_tag}"
                f"{self.LSTM_NUM_LAYERS}xLSTM"
                f"{self.LSTM_HIDDEN_SIZE}"
                f"_FC{TARGET_DIM}"
            )
        elif self.MODEL_TYPE == "mlp":
            self.MODEL_ARCHITECTURE_TAG = (
                f"MLP{INPUT_DIM}"
                f"x{self.MLP_HIDDEN_SIZE_1}"
                f"x{self.MLP_HIDDEN_SIZE_2}"
                f"x{TARGET_DIM}"
            )
        else:
            self.MODEL_ARCHITECTURE_TAG = self.MODEL_TYPE

        project_root = Path(self.PROJECT_ROOT)

        self.H5_DATA_PATH = str(
            project_root
            / self.STEP08_ROOT_NAME
            / self.DATASET_RUN_NAME
            / f"multiplier_{self.MULTIPLIER:02d}"
            / f"{self.PHASE}.h5"
        )

        self.LCS_BASELINE_H5_PATH = str(
            project_root
            / self.STEP09_ROOT_NAME
            / f"{self.PHASE}_multiplier_{self.MULTIPLIER:02d}_LCS_baseline.h5"
        )

        experiment_name = (
            f"{self.DATASET_RUN_NAME}"
            f"__{self.PHASE}"
            f"__m{self.MULTIPLIER:02d}"
            f"__{self.MODEL_TYPE}"
            f"__{self.MODEL_ARCHITECTURE_TAG}"
        )

        experiment_root = (
            project_root
            / self.STEP10_ROOT_NAME
            / experiment_name
        )

        self.EXPERIMENT_ROOT = str(experiment_root)
        self.CHECKPOINT_DIR = str(experiment_root / "checkpoints")
        self.LOG_DIR = str(experiment_root / "logs")
        self.FIGURE_DIR = str(experiment_root / "figures")
        self.EXPORT_DIR = str(experiment_root / "exports")

        self.NORMALIZATION_PATH = str(
            experiment_root
            / "normalization_train_only.npz"
        )

        self.HISTORY_CSV_PATH = str(
            experiment_root
            / "training_history.csv"
        )

        self.CONFIG_SNAPSHOT_PATH = str(
            experiment_root
            / "config_snapshot.json"
        )

        self.LAST_CHECKPOINT_PATH = str(
            experiment_root
            / "checkpoints"
            / "last_checkpoint.pt"
        )

        self.BEST_CHECKPOINT_PATH = str(
            experiment_root
            / "checkpoints"
            / "best_checkpoint.pt"
        )

        self.LIVE_DASHBOARD_PATH = str(
            experiment_root
            / "figures"
            / "live_training_dashboard.png"
        )

        self.TRAINING_SUMMARY_PATH = str(
            experiment_root
            / "training_summary.json"
        )

        self.validate()

    # -------------------------------------------------------------------------
    # Convenience properties
    # -------------------------------------------------------------------------

    @property
    def model_display_name(self) -> str:
        return MODEL_DISPLAY_NAMES[self.MODEL_TYPE]

    @property
    def is_lstm(self) -> bool:
        return self.MODEL_TYPE == "lstm_fc"

    @property
    def is_mlp(self) -> bool:
        return self.MODEL_TYPE == "mlp"

    @property
    def is_baseline(self) -> bool:
        return self.MODEL_TYPE in ("linear", "ridge", "lasso")

    @property
    def expected_sequence_length(self) -> int:
        if self.BASE_SEQUENCE_LENGTH % self.MULTIPLIER != 0:
            raise ValueError(
                f"Base length {self.BASE_SEQUENCE_LENGTH} is not divisible by multiplier {self.MULTIPLIER}"
            )
        return self.BASE_SEQUENCE_LENGTH // self.MULTIPLIER

    @property
    def phase_h5_path(self) -> Path:
        return Path(self.H5_DATA_PATH)

    @property
    def lcs_h5_path(self) -> Path:
        return Path(self.LCS_BASELINE_H5_PATH)

    @property
    def experiment_root_path(self) -> Path:
        return Path(self.EXPERIMENT_ROOT)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self) -> None:
        if self.PHASE not in ALLOWED_PHASES:
            raise ValueError(
                f"PHASE must be one of {ALLOWED_PHASES}, got {self.PHASE!r}"
            )

        if self.MODEL_TYPE not in ALLOWED_MODEL_TYPES:
            raise ValueError(
                f"MODEL_TYPE must be one of {ALLOWED_MODEL_TYPES}, "
                f"got {self.MODEL_TYPE!r}"
            )

        if self.MULTIPLIER <= 0:
            raise ValueError("MULTIPLIER must be positive")

        if self.BASE_SEQUENCE_LENGTH <= 0:
            raise ValueError("BASE_SEQUENCE_LENGTH must be positive")

        if self.BASE_SEQUENCE_LENGTH % self.MULTIPLIER != 0:
            raise ValueError(
                f"MULTIPLIER={self.MULTIPLIER} does not divide base length "
                f"{self.BASE_SEQUENCE_LENGTH}"
            )

        if INPUT_DIM != len(INPUT_ORDER):
            raise RuntimeError(
                f"INPUT_DIM={INPUT_DIM}, but INPUT_ORDER has {len(INPUT_ORDER)} entries"
            )

        if TARGET_DIM != len(TARGET_ORDER):
            raise RuntimeError(
                f"TARGET_DIM={TARGET_DIM}, but TARGET_ORDER has "
                f"{len(TARGET_ORDER)} entries"
            )

        if self.DATA_ACCESS_MODE not in ("ram", "h5"):
            raise ValueError("DATA_ACCESS_MODE must be 'ram' or 'h5'")

        if self.NORMALIZATION_MODE != "standard":
            raise ValueError("Only NORMALIZATION_MODE='standard' is supported")

        if self.NORMALIZATION_EPS <= 0.0:
            raise ValueError("NORMALIZATION_EPS must be positive")

        if self.NORMALIZATION_CHUNK_TRAJECTORIES <= 0:
            raise ValueError(
                "NORMALIZATION_CHUNK_TRAJECTORIES must be positive"
            )

        if self.NORMALIZATION_USE_SAMPLE_WEIGHT:
            raise ValueError(
                "NORMALIZATION_USE_SAMPLE_WEIGHT must remain False because "
                "Sample_Weight is reserved for loss/metrics"
            )

        for name, value in (
            ("MAX_TRAIN_TRAJECTORIES", self.MAX_TRAIN_TRAJECTORIES),
            ("MAX_VAL_TRAJECTORIES", self.MAX_VAL_TRAJECTORIES),
        ):
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive or None")

        if self.BATCH_SIZE <= 0:
            raise ValueError("BATCH_SIZE must be positive")

        if self.NUM_WORKERS < 0:
            raise ValueError("NUM_WORKERS cannot be negative")

        if self.PERSISTENT_WORKERS and self.NUM_WORKERS == 0:
            raise ValueError(
                "PERSISTENT_WORKERS=True requires NUM_WORKERS > 0"
            )

        if self.MAX_EPOCHS <= 0:
            raise ValueError("MAX_EPOCHS must be positive")

        if self.LEARNING_RATE <= 0.0:
            raise ValueError("LEARNING_RATE must be positive")

        if self.EARLY_STOPPING_PATIENCE <= 0:
            raise ValueError("EARLY_STOPPING_PATIENCE must be positive")

        if self.LSTM_HIDDEN_SIZE <= 0:
            raise ValueError("LSTM_HIDDEN_SIZE must be positive")

        if self.LSTM_NUM_LAYERS <= 0:
            raise ValueError("LSTM_NUM_LAYERS must be positive")

        if not 0.0 <= self.LSTM_DROPOUT < 1.0:
            raise ValueError("LSTM_DROPOUT must be in [0, 1)")

        if self.LSTM_NUM_LAYERS == 1 and self.LSTM_DROPOUT != 0.0:
            # PyTorch ignores recurrent dropout for one layer. Do not silently
            # leave a misleading configuration.
            raise ValueError(
                "Set LSTM_DROPOUT=0.0 when LSTM_NUM_LAYERS=1"
            )

        if self.LSTM_OUTPUT_MODE != "absolute_stress":
            raise ValueError(
                "Only LSTM_OUTPUT_MODE='absolute_stress' is currently supported"
            )

        if self.MLP_HIDDEN_SIZE_1 <= 0:
            raise ValueError("MLP_HIDDEN_SIZE_1 must be positive")

        if self.MLP_HIDDEN_SIZE_2 <= 0:
            raise ValueError("MLP_HIDDEN_SIZE_2 must be positive")

        if self.MLP_ACTIVATION != "gelu":
            raise ValueError("Only MLP_ACTIVATION='gelu' is currently supported")

        if not 0.0 <= self.MLP_DROPOUT < 1.0:
            raise ValueError("MLP_DROPOUT must be in [0, 1)")

        if self.MLP_OUTPUT_MODE != "absolute_stress":
            raise ValueError(
                "Only MLP_OUTPUT_MODE='absolute_stress' is currently supported"
            )

        if self.MLP_WEIGHT_DECAY < 0.0:
            raise ValueError("MLP_WEIGHT_DECAY cannot be negative")

        if self.OPTIMIZER not in ("adam", "adamw"):
            raise ValueError("OPTIMIZER must be adam or adamw")

        if self.LR_SCHEDULER not in ("none", "reduce_on_plateau"):
            raise ValueError(
                "LR_SCHEDULER must be none or reduce_on_plateau"
            )

        if not 0.0 < self.LR_REDUCE_FACTOR < 1.0:
            raise ValueError("LR_REDUCE_FACTOR must be in (0, 1)")

        if self.LR_REDUCE_PATIENCE < 0:
            raise ValueError("LR_REDUCE_PATIENCE cannot be negative")

        if self.MIN_LEARNING_RATE <= 0.0:
            raise ValueError("MIN_LEARNING_RATE must be positive")

        if self.MIN_LEARNING_RATE > self.LEARNING_RATE:
            raise ValueError(
                "MIN_LEARNING_RATE cannot exceed LEARNING_RATE"
            )

        if self.WEIGHT_DECAY < 0.0:
            raise ValueError("WEIGHT_DECAY cannot be negative")

        if self.EARLY_STOPPING_MIN_DELTA < 0.0:
            raise ValueError("EARLY_STOPPING_MIN_DELTA cannot be negative")

        if self.GRADIENT_CLIP_NORM < 0.0:
            raise ValueError("GRADIENT_CLIP_NORM cannot be negative")

        if self.DATA_LOSS != "weighted_mse":
            raise ValueError(
                "Only DATA_LOSS='weighted_mse' is supported in this series"
            )

        if self.BASELINE_INPUT_MODE != "current_step_only":
            raise ValueError(
                "Only BASELINE_INPUT_MODE='current_step_only' is supported"
            )

        if self.BASELINE_POINT_CHUNK_SIZE <= 0:
            raise ValueError("BASELINE_POINT_CHUNK_SIZE must be positive")

        if self.RIDGE_ALPHA < 0.0:
            raise ValueError("RIDGE_ALPHA cannot be negative")

        if self.LASSO_ALPHA <= 0.0:
            raise ValueError("LASSO_ALPHA must be positive")

        if self.LASSO_MAX_ITER <= 0:
            raise ValueError("LASSO_MAX_ITER must be positive")

        if self.LASSO_TOL <= 0.0:
            raise ValueError("LASSO_TOL must be positive")

        if self.LASSO_SELECTION not in ("cyclic", "random"):
            raise ValueError(
                "LASSO_SELECTION must be 'cyclic' or 'random'"
            )

        for name, value in (
            ("BASELINE_MAX_TRAIN_POINTS", self.BASELINE_MAX_TRAIN_POINTS),
            ("BASELINE_MAX_VAL_POINTS", self.BASELINE_MAX_VAL_POINTS),
        ):
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive or None")

        if self.VALIDATION_METRIC != "weighted_global_relative_l2_pct":
            raise ValueError(
                "VALIDATION_METRIC must match Step-09: "
                "'weighted_global_relative_l2_pct'"
            )

        if not self.EXCLUDE_FIRST_STEP_FOR_LCS_COMPARISON:
            raise ValueError(
                "Strict Step-09 comparison requires "
                "EXCLUDE_FIRST_STEP_FOR_LCS_COMPARISON=True"
            )

        if self.RESUME_MODE not in ("auto", "never", "required"):
            raise ValueError(
                "RESUME_MODE must be auto, never or required"
            )

        if self.SAVE_EVERY_N_EPOCHS < 0:
            raise ValueError("SAVE_EVERY_N_EPOCHS cannot be negative")

        if self.DASHBOARD_Y_SCALE not in ("linear", "log"):
            raise ValueError("DASHBOARD_Y_SCALE must be linear or log")

        if self.DASHBOARD_DPI <= 0:
            raise ValueError("DASHBOARD_DPI must be positive")

        if self.DASHBOARD_REFRESH_EVERY_EPOCHS <= 0:
            raise ValueError(
                "DASHBOARD_REFRESH_EVERY_EPOCHS must be positive"
            )

        if self.SAVE_DASHBOARD_EVERY_EPOCHS <= 0:
            raise ValueError(
                "SAVE_DASHBOARD_EVERY_EPOCHS must be positive"
            )

        if self.LIVE_TABLE_ROWS <= 0:
            raise ValueError("LIVE_TABLE_ROWS must be positive")

        if self.PRINT_EVERY_EPOCHS <= 0:
            raise ValueError("PRINT_EVERY_EPOCHS must be positive")

        if self.EVALUATE_EVERY_EPOCHS <= 0:
            raise ValueError("EVALUATE_EVERY_EPOCHS must be positive")

        try:
            torch.device(self.DEVICE)
        except Exception as exc:
            raise ValueError(
                f"Invalid PyTorch DEVICE={self.DEVICE!r}"
            ) from exc

    # -------------------------------------------------------------------------
    # Runtime helpers
    # -------------------------------------------------------------------------

    def create_output_directories(self) -> None:
        for raw in (
            self.EXPERIMENT_ROOT,
            self.CHECKPOINT_DIR,
            self.LOG_DIR,
            self.FIGURE_DIR,
            self.EXPORT_DIR,
        ):
            Path(raw).mkdir(parents=True, exist_ok=True)

    def validate_required_files(self) -> None:
        if not self.phase_h5_path.is_file():
            raise FileNotFoundError(
                f"Step-08 H5 was not found: {self.phase_h5_path}"
            )

        if self.REQUIRE_LCS_REFERENCE and not self.lcs_h5_path.is_file():
            raise FileNotFoundError(
                f"Step-09 LCS baseline H5 was not found: {self.lcs_h5_path}"
            )

    def snapshot_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["MODEL_DISPLAY_NAME"] = self.model_display_name
        data["EXPECTED_SEQUENCE_LENGTH"] = self.expected_sequence_length
        data["INPUT_DIM"] = INPUT_DIM
        data["TARGET_DIM"] = TARGET_DIM
        data["INPUT_ORDER"] = INPUT_ORDER
        data["TARGET_ORDER"] = TARGET_ORDER
        return data

    def save_snapshot(self) -> None:
        self.create_output_directories()
        Path(self.CONFIG_SNAPSHOT_PATH).write_text(
            json.dumps(
                self.snapshot_dict(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def print_summary(self) -> None:
        print("=" * 108)
        print("Step-10 training configuration")
        print("Phase              :", self.PHASE)
        print("Multiplier         :", self.MULTIPLIER)
        print("Sequence length    :", self.expected_sequence_length)
        print("Model              :", self.model_display_name)
        print("Model key          :", self.MODEL_TYPE)
        print("Architecture       :", self.MODEL_ARCHITECTURE_TAG)
        print("Input dimension    :", INPUT_DIM)
        print("Target dimension   :", TARGET_DIM)
        print("Data access mode   :", self.DATA_ACCESS_MODE)
        print("Step-08 H5         :", self.H5_DATA_PATH)
        print("Step-09 LCS H5     :", self.LCS_BASELINE_H5_PATH)
        print("Experiment root    :", self.EXPERIMENT_ROOT)
        print("Normalization      :", self.NORMALIZATION_PATH)
        print("Resume mode        :", self.RESUME_MODE)
        print("Last checkpoint    :", self.LAST_CHECKPOINT_PATH)
        print("Device             :", self.DEVICE)
        print("Train/val split    : reuse Step-08 indices")
        print("Physics loss       : disabled")
        print("LCS comparison     : val weighted global relative L2, first step excluded")
        if self.is_baseline:
            print("Baseline history   : none; current-step features only")
        elif self.is_mlp:
            print("MLP history        : none; each time point is processed independently")
        else:
            print("LSTM history       : full sequence")
        print("=" * 108)


# =============================================================================
# 2. Global configuration instance
# =============================================================================

CFG = Config()


# =============================================================================
# 3. Reproducibility utility
# =============================================================================

def set_global_seed(cfg: Config = CFG) -> None:
    seed = int(cfg.RANDOM_SEED)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cfg.DETERMINISTIC_TORCH:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


# =============================================================================
# 4. Standalone check
# =============================================================================

if __name__ == "__main__":
    CFG.print_summary()
    CFG.validate_required_files()
    CFG.create_output_directories()
    CFG.save_snapshot()
    set_global_seed(CFG)

    print("Configuration check: PASS")
    print("Snapshot:", CFG.CONFIG_SNAPSHOT_PATH)
