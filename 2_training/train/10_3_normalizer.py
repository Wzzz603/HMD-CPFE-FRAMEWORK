#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_3_normalizer.py

Compile and reuse train-only standardization statistics for Step-10.

The same normalization file is shared by:
    LSTM + FC
    Linear Regression
    Ridge Regression
    Lasso Regression

for one fixed:
    dataset run + phase + increment multiplier.

Rules
-----
1. Statistics are computed from Step-08 TRAIN trajectories only.
2. All time points in each selected train trajectory are included.
3. Sample_Weight is NOT used to calculate mean/std. It remains a loss weight.
4. The 22 input features use the exact order fixed by Step-08 and 10_1_config.
5. Targets use the six stress components in their exact fixed order.
6. Validation data never contributes to normalization.
7. The original Step-08 H5 is never modified.
8. Computation is streamed in trajectory chunks; the full H5 need not fit RAM.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch


# =============================================================================
# 0. Dynamic import of 10_2_h5_dataset.py
# =============================================================================

_THIS_DIR = Path(__file__).resolve().parent
_DATASET_PATH = _THIS_DIR / "10_2_h5_dataset.py"

if not _DATASET_PATH.is_file():
    raise FileNotFoundError(
        "10_2_h5_dataset.py must be in the same directory: %s"
        % _DATASET_PATH
    )

_spec = importlib.util.spec_from_file_location(
    "step10_h5_dataset",
    _DATASET_PATH,
)
if _spec is None or _spec.loader is None:
    raise ImportError(
        "Could not create import specification for %s" % _DATASET_PATH
    )

_dataset_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _dataset_module
_spec.loader.exec_module(_dataset_module)

CFG = _dataset_module.CFG
INPUT_DIM = int(_dataset_module.INPUT_DIM)
TARGET_DIM = int(_dataset_module.TARGET_DIM)
INPUT_ORDER = tuple(_dataset_module.INPUT_ORDER)
TARGET_ORDER = tuple(_dataset_module.TARGET_ORDER)

PhaseH5Info = _dataset_module.PhaseH5Info
PhaseH5Store = _dataset_module.PhaseH5Store
inspect_phase_h5 = _dataset_module.inspect_phase_h5
assemble_model_inputs_numpy = _dataset_module.assemble_model_inputs_numpy
_load_split_indices = _dataset_module._load_split_indices


# =============================================================================
# 1. Running moments
# =============================================================================

@dataclass
class RunningMoments:
    """
    Numerically stable parallel/Welford moments for feature vectors.

    count is the number of rows observed. Population variance M2/count is used
    because these statistics describe the complete selected training dataset.
    """
    feature_dim: int

    def __post_init__(self) -> None:
        if int(self.feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        self.count: int = 0
        self.mean: np.ndarray = np.zeros(
            int(self.feature_dim),
            dtype=np.float64,
        )
        self.m2: np.ndarray = np.zeros(
            int(self.feature_dim),
            dtype=np.float64,
        )
        self.minimum: np.ndarray = np.full(
            int(self.feature_dim),
            np.inf,
            dtype=np.float64,
        )
        self.maximum: np.ndarray = np.full(
            int(self.feature_dim),
            -np.inf,
            dtype=np.float64,
        )

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values)
        if values.shape[-1] != self.feature_dim:
            raise ValueError(
                "Last dimension=%d, expected=%d"
                % (values.shape[-1], self.feature_dim)
            )

        flat = np.asarray(
            values.reshape(-1, self.feature_dim),
            dtype=np.float64,
        )
        batch_count = int(flat.shape[0])
        if batch_count == 0:
            return

        if not np.isfinite(flat).all():
            raise ValueError("NaN/Inf encountered while computing moments")

        batch_mean = np.mean(flat, axis=0, dtype=np.float64)
        centered = flat - batch_mean
        batch_m2 = np.sum(
            centered * centered,
            axis=0,
            dtype=np.float64,
        )
        batch_min = np.min(flat, axis=0)
        batch_max = np.max(flat, axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            self.minimum = batch_min
            self.maximum = batch_max
            return

        old_count = self.count
        new_count = old_count + batch_count
        delta = batch_mean - self.mean

        self.mean = (
            self.mean
            + delta * (batch_count / float(new_count))
        )
        self.m2 = (
            self.m2
            + batch_m2
            + delta * delta
            * (old_count * batch_count / float(new_count))
        )
        self.minimum = np.minimum(self.minimum, batch_min)
        self.maximum = np.maximum(self.maximum, batch_max)
        self.count = new_count

    def finalize(
        self,
        epsilon: float,
    ) -> Dict[str, np.ndarray | int]:
        if self.count <= 0:
            raise RuntimeError("No samples accumulated")

        variance = self.m2 / float(self.count)
        # Tiny negative values may occur from floating-point roundoff.
        variance = np.maximum(variance, 0.0)
        raw_std = np.sqrt(variance)

        constant_mask = raw_std < float(epsilon)
        safe_std = raw_std.copy()
        safe_std[constant_mask] = 1.0

        return {
            "count": int(self.count),
            "mean": np.asarray(self.mean, dtype=np.float64),
            "raw_std": np.asarray(raw_std, dtype=np.float64),
            "safe_std": np.asarray(safe_std, dtype=np.float64),
            "constant_mask": np.asarray(constant_mask, dtype=np.bool_),
            "minimum": np.asarray(self.minimum, dtype=np.float64),
            "maximum": np.asarray(self.maximum, dtype=np.float64),
        }


# =============================================================================
# 2. Saved normalization object
# =============================================================================

@dataclass(frozen=True)
class NormalizationStats:
    path: Path
    schema_name: str
    source_h5: Path
    source_h5_size_bytes: int
    source_h5_modified_time_ns: int
    train_indices_sha256: str
    normalization_epsilon: float
    phase: str
    multiplier: int
    sequence_length: int
    train_trajectory_count: int
    input_observation_count: int
    target_observation_count: int

    input_names: Tuple[str, ...]
    target_names: Tuple[str, ...]

    input_mean: np.ndarray
    input_std: np.ndarray
    input_raw_std: np.ndarray
    input_constant_mask: np.ndarray
    input_min: np.ndarray
    input_max: np.ndarray

    target_mean: np.ndarray
    target_std: np.ndarray
    target_raw_std: np.ndarray
    target_constant_mask: np.ndarray
    target_min: np.ndarray
    target_max: np.ndarray

    use_sample_weight_for_stats: bool
    created_time: str

    def normalize_input_numpy(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return np.asarray(
            (x - self.input_mean.astype(np.float32))
            / self.input_std.astype(np.float32),
            dtype=np.float32,
        )

    def denormalize_input_numpy(self, x_norm: np.ndarray) -> np.ndarray:
        x_norm = np.asarray(x_norm, dtype=np.float32)
        return np.asarray(
            x_norm * self.input_std.astype(np.float32)
            + self.input_mean.astype(np.float32),
            dtype=np.float32,
        )

    def normalize_target_numpy(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        return np.asarray(
            (y - self.target_mean.astype(np.float32))
            / self.target_std.astype(np.float32),
            dtype=np.float32,
        )

    def denormalize_target_numpy(self, y_norm: np.ndarray) -> np.ndarray:
        y_norm = np.asarray(y_norm, dtype=np.float32)
        return np.asarray(
            y_norm * self.target_std.astype(np.float32)
            + self.target_mean.astype(np.float32),
            dtype=np.float32,
        )

    def validate_against_config(self, cfg=CFG) -> None:
        if self.schema_name != "CPFE_STEP10_NORMALIZATION_V2":
            raise ValueError(
                "Unsupported normalization schema=%r; expected V2"
                % self.schema_name
            )
        if self.phase != str(cfg.PHASE).upper():
            raise ValueError(
                "Normalization phase=%r, config phase=%r"
                % (self.phase, cfg.PHASE)
            )
        if self.multiplier != int(cfg.MULTIPLIER):
            raise ValueError(
                "Normalization multiplier=%d, config=%d"
                % (self.multiplier, cfg.MULTIPLIER)
            )
        if self.sequence_length != int(cfg.expected_sequence_length):
            raise ValueError(
                "Normalization sequence length=%d, config=%d"
                % (
                    self.sequence_length,
                    cfg.expected_sequence_length,
                )
            )
        if self.input_names != INPUT_ORDER:
            raise ValueError("Normalization input order does not match config")
        if self.target_names != TARGET_ORDER:
            raise ValueError("Normalization target order does not match config")
        if self.use_sample_weight_for_stats:
            raise ValueError(
                "This training series requires unweighted normalization stats"
            )
        if self.input_mean.shape != (INPUT_DIM,):
            raise ValueError("Invalid input_mean shape")
        if self.input_std.shape != (INPUT_DIM,):
            raise ValueError("Invalid input_std shape")
        if self.target_mean.shape != (TARGET_DIM,):
            raise ValueError("Invalid target_mean shape")
        if self.target_std.shape != (TARGET_DIM,):
            raise ValueError("Invalid target_std shape")
        if not np.isfinite(self.input_mean).all():
            raise ValueError("Non-finite input_mean")
        if not np.isfinite(self.input_std).all():
            raise ValueError("Non-finite input_std")
        if not np.isfinite(self.target_mean).all():
            raise ValueError("Non-finite target_mean")
        if not np.isfinite(self.target_std).all():
            raise ValueError("Non-finite target_std")
        if np.any(self.input_std <= 0.0):
            raise ValueError("input_std must be positive")
        if np.any(self.target_std <= 0.0):
            raise ValueError("target_std must be positive")
        if not math.isfinite(self.normalization_epsilon):
            raise ValueError("normalization_epsilon must be finite")
        if self.normalization_epsilon <= 0.0:
            raise ValueError("normalization_epsilon must be positive")
        if not math.isclose(
            self.normalization_epsilon,
            float(cfg.NORMALIZATION_EPS),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                "Normalization epsilon=%r, config epsilon=%r"
                % (self.normalization_epsilon, cfg.NORMALIZATION_EPS)
            )
        if self.source_h5_size_bytes <= 0:
            raise ValueError("Invalid saved source H5 size")
        if self.source_h5_modified_time_ns <= 0:
            raise ValueError("Invalid saved source H5 modification time")
        if len(self.train_indices_sha256) != 64:
            raise ValueError("Invalid train-index SHA256 digest")


class TorchNormalizer:
    """
    Device-aware tensor normalization used later by the trainer and evaluator.
    """

    def __init__(
        self,
        stats: NormalizationStats,
        device: torch.device | str,
    ) -> None:
        self.stats = stats
        self.device = torch.device(device)

        self.input_mean = torch.as_tensor(
            stats.input_mean,
            dtype=torch.float32,
            device=self.device,
        )
        self.input_std = torch.as_tensor(
            stats.input_std,
            dtype=torch.float32,
            device=self.device,
        )
        self.target_mean = torch.as_tensor(
            stats.target_mean,
            dtype=torch.float32,
            device=self.device,
        )
        self.target_std = torch.as_tensor(
            stats.target_std,
            dtype=torch.float32,
            device=self.device,
        )

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mean) / self.input_std

    def denormalize_input(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.input_std + self.input_mean

    def normalize_target(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.target_mean) / self.target_std

    def denormalize_target(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.target_std + self.target_mean


# =============================================================================
# 3. Compilation
# =============================================================================

def _now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _source_signature(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "modified_time_ns": int(stat.st_mtime_ns),
    }


def _indices_sha256(indices: np.ndarray) -> str:
    indices = np.ascontiguousarray(
        np.asarray(indices, dtype=np.int64)
    )
    if indices.ndim != 1:
        raise ValueError("Train indices must be one-dimensional")
    return hashlib.sha256(indices.tobytes(order="C")).hexdigest()


def _validate_saved_source_and_selection(
    stats: NormalizationStats,
    cfg=CFG,
) -> None:
    configured_source = Path(cfg.H5_DATA_PATH).resolve()
    saved_source = stats.source_h5.resolve()
    if saved_source != configured_source:
        raise ValueError(
            "Normalization source H5 differs from configured H5:\n"
            "saved:      %s\nconfigured: %s"
            % (saved_source, configured_source)
        )

    current_signature = _source_signature(configured_source)
    if int(current_signature["size_bytes"]) != int(
        stats.source_h5_size_bytes
    ):
        raise ValueError(
            "Step-08 H5 size changed after normalization was created: "
            "saved=%d, current=%d"
            % (
                stats.source_h5_size_bytes,
                current_signature["size_bytes"],
            )
        )
    if int(current_signature["modified_time_ns"]) != int(
        stats.source_h5_modified_time_ns
    ):
        raise ValueError(
            "Step-08 H5 modification time changed after normalization "
            "was created"
        )

    info = inspect_phase_h5(cfg)
    train_indices = _load_split_indices(
        info=info,
        split_name="train",
        max_count=cfg.MAX_TRAIN_TRAJECTORIES,
        random_seed=cfg.RANDOM_SEED,
    )
    current_digest = _indices_sha256(train_indices)
    if current_digest != stats.train_indices_sha256:
        raise ValueError(
            "Selected Step-08 train indices differ from the normalization "
            "file"
        )

    expected_train_count = int(train_indices.size)
    expected_observation_count = int(
        expected_train_count * info.sequence_length
    )
    if stats.train_trajectory_count != expected_train_count:
        raise ValueError(
            "Normalization train trajectory count=%d, current selection=%d"
            % (stats.train_trajectory_count, expected_train_count)
        )
    if stats.input_observation_count != expected_observation_count:
        raise ValueError(
            "Normalization input observation count=%d, expected=%d"
            % (stats.input_observation_count, expected_observation_count)
        )
    if stats.target_observation_count != expected_observation_count:
        raise ValueError(
            "Normalization target observation count=%d, expected=%d"
            % (stats.target_observation_count, expected_observation_count)
        )


def compile_train_only_normalization(
    cfg=CFG,
    force_recompute: Optional[bool] = None,
) -> NormalizationStats:
    """
    Stream the fixed Step-08 train split and save one shared .npz file.
    """
    cfg.validate_required_files()
    cfg.create_output_directories()

    output_path = Path(cfg.NORMALIZATION_PATH)
    if force_recompute is None:
        force_recompute = bool(cfg.NORMALIZATION_OVERWRITE)

    if output_path.exists() and not force_recompute:
        try:
            stats = load_normalization_stats(output_path, cfg)
        except Exception as exc:
            print(
                "[NORMALIZATION] Existing file is incompatible and will be "
                "recomputed: %s" % output_path
            )
            print("[NORMALIZATION] Reason: %s" % exc)
        else:
            print(
                "[NORMALIZATION] Existing compatible file reused: %s"
                % output_path
            )
            return stats

    info: PhaseH5Info = inspect_phase_h5(cfg)

    if bool(cfg.NORMALIZATION_USE_SAMPLE_WEIGHT):
        raise ValueError(
            "NORMALIZATION_USE_SAMPLE_WEIGHT must remain False"
        )

    train_indices = _load_split_indices(
        info=info,
        split_name="train",
        max_count=cfg.MAX_TRAIN_TRAJECTORIES,
        random_seed=cfg.RANDOM_SEED,
    )
    train_indices_sha256 = _indices_sha256(train_indices)

    chunk_size = int(cfg.NORMALIZATION_CHUNK_TRAJECTORIES)
    x_moments = RunningMoments(INPUT_DIM)
    y_moments = RunningMoments(TARGET_DIM)

    store = PhaseH5Store(
        info=info,
        access_mode="h5",
    )

    started = time.perf_counter()
    try:
        for start in range(0, train_indices.size, chunk_size):
            end = min(start + chunk_size, train_indices.size)
            source_indices = train_indices[start:end]
            arrays = store.read_core_rows(source_indices)

            x = assemble_model_inputs_numpy(
                F_end=arrays["F_end"],
                theta_start=arrays["theta_start"],
                static_values=arrays["static"],
                info=info,
            )
            y = np.asarray(
                arrays["sigma_end"],
                dtype=np.float32,
            )

            if cfg.FAIL_ON_NAN_OR_INF:
                if not np.isfinite(x).all():
                    raise ValueError(
                        "NaN/Inf in assembled train inputs [%d:%d]"
                        % (start, end)
                    )
                if not np.isfinite(y).all():
                    raise ValueError(
                        "NaN/Inf in train targets [%d:%d]"
                        % (start, end)
                    )

            x_moments.update(x)
            y_moments.update(y)

            processed = end
            if (
                processed == train_indices.size
                or processed % (chunk_size * 20) == 0
            ):
                elapsed = time.perf_counter() - started
                rate = processed / max(elapsed, 1.0e-12)
                print(
                    "[NORMALIZATION] %d / %d train trajectories "
                    "(%.1f%%), %.0f traj/s"
                    % (
                        processed,
                        train_indices.size,
                        100.0 * processed / train_indices.size,
                        rate,
                    )
                )

            del arrays, x, y
    finally:
        store.close()

    x_final = x_moments.finalize(cfg.NORMALIZATION_EPS)
    y_final = y_moments.finalize(cfg.NORMALIZATION_EPS)

    expected_observations = int(
        train_indices.size * info.sequence_length
    )
    if int(x_final["count"]) != expected_observations:
        raise RuntimeError(
            "Input observation count=%d, expected=%d"
            % (x_final["count"], expected_observations)
        )
    if int(y_final["count"]) != expected_observations:
        raise RuntimeError(
            "Target observation count=%d, expected=%d"
            % (y_final["count"], expected_observations)
        )

    source_signature = _source_signature(info.path)
    created_time = _now_string()

    metadata = {
        "schema_name": "CPFE_STEP10_NORMALIZATION_V2",
        "created_time": created_time,
        "source_h5": source_signature,
        "phase": info.phase,
        "increment_multiplier": int(info.multiplier),
        "sequence_length": int(info.sequence_length),
        "train_trajectory_count": int(train_indices.size),
        "train_indices_sha256": train_indices_sha256,
        "max_train_trajectories": (
            None
            if cfg.MAX_TRAIN_TRAJECTORIES is None
            else int(cfg.MAX_TRAIN_TRAJECTORIES)
        ),
        "random_seed_for_debug_selection": int(cfg.RANDOM_SEED),
        "input_observation_count": int(x_final["count"]),
        "target_observation_count": int(y_final["count"]),
        "input_order": list(INPUT_ORDER),
        "target_order": list(TARGET_ORDER),
        "normalization_mode": "standard",
        "variance_definition": "population_variance_m2_div_count",
        "normalization_epsilon": float(cfg.NORMALIZATION_EPS),
        "sample_weight_used_for_stats": False,
        "validation_used_for_stats": False,
        "first_time_point_included": True,
        "constant_feature_policy": "safe_std_set_to_1",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )
    if temporary_path.exists():
        temporary_path.unlink()

    with temporary_path.open("wb") as f:
        np.savez_compressed(
            f,
            metadata_json=np.asarray(
                json.dumps(metadata, ensure_ascii=False),
            ),
            input_names=np.asarray(INPUT_ORDER, dtype="U64"),
            target_names=np.asarray(TARGET_ORDER, dtype="U64"),

            input_mean=np.asarray(x_final["mean"], dtype=np.float64),
            input_std=np.asarray(x_final["safe_std"], dtype=np.float64),
            input_raw_std=np.asarray(
                x_final["raw_std"],
                dtype=np.float64,
            ),
            input_constant_mask=np.asarray(
                x_final["constant_mask"],
                dtype=np.bool_,
            ),
            input_min=np.asarray(
                x_final["minimum"],
                dtype=np.float64,
            ),
            input_max=np.asarray(
                x_final["maximum"],
                dtype=np.float64,
            ),

            target_mean=np.asarray(y_final["mean"], dtype=np.float64),
            target_std=np.asarray(y_final["safe_std"], dtype=np.float64),
            target_raw_std=np.asarray(
                y_final["raw_std"],
                dtype=np.float64,
            ),
            target_constant_mask=np.asarray(
                y_final["constant_mask"],
                dtype=np.bool_,
            ),
            target_min=np.asarray(
                y_final["minimum"],
                dtype=np.float64,
            ),
            target_max=np.asarray(
                y_final["maximum"],
                dtype=np.float64,
            ),
        )

    os.replace(temporary_path, output_path)

    sidecar_json = output_path.with_suffix(".json")
    sidecar_json.write_text(
        json.dumps(
            {
                **metadata,
                "normalization_npz": str(output_path),
                "input_constant_features": [
                    INPUT_ORDER[i]
                    for i, flag in enumerate(
                        np.asarray(
                            x_final["constant_mask"],
                            dtype=np.bool_,
                        )
                    )
                    if bool(flag)
                ],
                "target_constant_features": [
                    TARGET_ORDER[i]
                    for i, flag in enumerate(
                        np.asarray(
                            y_final["constant_mask"],
                            dtype=np.bool_,
                        )
                    )
                    if bool(flag)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    stats = load_normalization_stats(output_path, cfg)

    elapsed = time.perf_counter() - started
    print("=" * 108)
    print("Train-only normalization compiled")
    print("Phase                :", stats.phase)
    print("Multiplier           :", stats.multiplier)
    print("Sequence length      :", stats.sequence_length)
    print("Train trajectories   :", stats.train_trajectory_count)
    print("Observations         :", stats.input_observation_count)
    print("Sample-weighted stats: NO")
    print("Validation used      : NO")
    print("Input constants      :", [
        stats.input_names[i]
        for i, flag in enumerate(stats.input_constant_mask)
        if bool(flag)
    ])
    print("Target constants     :", [
        stats.target_names[i]
        for i, flag in enumerate(stats.target_constant_mask)
        if bool(flag)
    ])
    print("Output NPZ           :", output_path)
    print("Output JSON          :", sidecar_json)
    print("Elapsed              : %.2f s" % elapsed)
    print("=" * 108)

    return stats


# =============================================================================
# 4. Loading and validation
# =============================================================================

def load_normalization_stats(
    path: Path | str | None = None,
    cfg=CFG,
) -> NormalizationStats:
    if path is None:
        path = Path(cfg.NORMALIZATION_PATH)
    else:
        path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            "Normalization file does not exist: %s" % path
        )

    with np.load(path, allow_pickle=False) as data:
        required = (
            "metadata_json",
            "input_names",
            "target_names",
            "input_mean",
            "input_std",
            "input_raw_std",
            "input_constant_mask",
            "input_min",
            "input_max",
            "target_mean",
            "target_std",
            "target_raw_std",
            "target_constant_mask",
            "target_min",
            "target_max",
        )
        for name in required:
            if name not in data:
                raise KeyError(
                    "Missing %s in normalization file %s"
                    % (name, path)
                )

        metadata_raw = data["metadata_json"]
        if metadata_raw.ndim == 0:
            metadata_text = str(metadata_raw.item())
        else:
            metadata_text = str(metadata_raw.reshape(-1)[0])
        metadata = json.loads(metadata_text)

        schema_name = str(metadata.get("schema_name", ""))
        if schema_name != "CPFE_STEP10_NORMALIZATION_V2":
            raise ValueError(
                "Normalization file uses unsupported or legacy schema=%r; "
                "it must be recomputed" % schema_name
            )

        source_metadata = metadata.get("source_h5")
        if not isinstance(source_metadata, dict):
            raise ValueError("Invalid source_h5 metadata")

        source_h5 = Path(source_metadata["path"])

        stats = NormalizationStats(
            path=path,
            schema_name=schema_name,
            source_h5=source_h5,
            source_h5_size_bytes=int(
                source_metadata["size_bytes"]
            ),
            source_h5_modified_time_ns=int(
                source_metadata["modified_time_ns"]
            ),
            train_indices_sha256=str(
                metadata["train_indices_sha256"]
            ),
            normalization_epsilon=float(
                metadata["normalization_epsilon"]
            ),
            phase=str(metadata["phase"]).upper(),
            multiplier=int(metadata["increment_multiplier"]),
            sequence_length=int(metadata["sequence_length"]),
            train_trajectory_count=int(
                metadata["train_trajectory_count"]
            ),
            input_observation_count=int(
                metadata["input_observation_count"]
            ),
            target_observation_count=int(
                metadata["target_observation_count"]
            ),
            input_names=tuple(
                str(x) for x in data["input_names"].tolist()
            ),
            target_names=tuple(
                str(x) for x in data["target_names"].tolist()
            ),

            input_mean=np.asarray(
                data["input_mean"],
                dtype=np.float64,
            ),
            input_std=np.asarray(
                data["input_std"],
                dtype=np.float64,
            ),
            input_raw_std=np.asarray(
                data["input_raw_std"],
                dtype=np.float64,
            ),
            input_constant_mask=np.asarray(
                data["input_constant_mask"],
                dtype=np.bool_,
            ),
            input_min=np.asarray(
                data["input_min"],
                dtype=np.float64,
            ),
            input_max=np.asarray(
                data["input_max"],
                dtype=np.float64,
            ),

            target_mean=np.asarray(
                data["target_mean"],
                dtype=np.float64,
            ),
            target_std=np.asarray(
                data["target_std"],
                dtype=np.float64,
            ),
            target_raw_std=np.asarray(
                data["target_raw_std"],
                dtype=np.float64,
            ),
            target_constant_mask=np.asarray(
                data["target_constant_mask"],
                dtype=np.bool_,
            ),
            target_min=np.asarray(
                data["target_min"],
                dtype=np.float64,
            ),
            target_max=np.asarray(
                data["target_max"],
                dtype=np.float64,
            ),

            use_sample_weight_for_stats=bool(
                metadata["sample_weight_used_for_stats"]
            ),
            created_time=str(metadata["created_time"]),
        )

    stats.validate_against_config(cfg)
    _validate_saved_source_and_selection(stats, cfg)
    return stats


# =============================================================================
# 5. Smoke test
# =============================================================================

def smoke_test_round_trip(
    stats: NormalizationStats,
    cfg=CFG,
) -> None:
    info = inspect_phase_h5(cfg)

    with h5py.File(info.path, "r") as h5:
        source_index = int(h5["split/train_indices"][0])
        F_end = np.asarray(
            h5["F_end"][source_index],
            dtype=np.float32,
        )
        theta = np.asarray(
            h5["theta_start"][source_index],
            dtype=np.float32,
        )
        static = np.asarray(
            h5["static"][source_index],
            dtype=np.float32,
        )
        y = np.asarray(
            h5["sigma_end"][source_index],
            dtype=np.float32,
        )

    x = assemble_model_inputs_numpy(
        F_end=F_end,
        theta_start=theta,
        static_values=static,
        info=info,
    )

    x_norm = stats.normalize_input_numpy(x)
    x_back = stats.denormalize_input_numpy(x_norm)

    y_norm = stats.normalize_target_numpy(y)
    y_back = stats.denormalize_target_numpy(y_norm)

    x_error = float(np.max(np.abs(x_back - x)))
    y_error = float(np.max(np.abs(y_back - y)))

    if not np.isfinite(x_norm).all():
        raise RuntimeError("Non-finite normalized x")
    if not np.isfinite(y_norm).all():
        raise RuntimeError("Non-finite normalized y")

    # Float32 round-trip tolerance scales with the physical magnitudes.
    x_tolerance = max(
        1.0e-5,
        1.0e-6 * float(np.max(np.abs(x)) + 1.0),
    )
    y_tolerance = max(
        1.0e-4,
        1.0e-6 * float(np.max(np.abs(y)) + 1.0),
    )

    if x_error > x_tolerance:
        raise RuntimeError(
            "Input round-trip max error %.6e exceeds %.6e"
            % (x_error, x_tolerance)
        )
    if y_error > y_tolerance:
        raise RuntimeError(
            "Target round-trip max error %.6e exceeds %.6e"
            % (y_error, y_tolerance)
        )

    torch_normalizer = TorchNormalizer(stats, device="cpu")
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)

    with torch.no_grad():
        x_torch_back = torch_normalizer.denormalize_input(
            torch_normalizer.normalize_input(x_tensor)
        )
        y_torch_back = torch_normalizer.denormalize_target(
            torch_normalizer.normalize_target(y_tensor)
        )

    if not torch.isfinite(x_torch_back).all():
        raise RuntimeError("Torch input round trip produced NaN/Inf")
    if not torch.isfinite(y_torch_back).all():
        raise RuntimeError("Torch target round trip produced NaN/Inf")

    print("Normalization smoke test: PASS")
    print("Input round-trip max abs error : %.6e" % x_error)
    print("Target round-trip max abs error: %.6e" % y_error)


# =============================================================================
# 6. Standalone entry
# =============================================================================

if __name__ == "__main__":
    CFG.print_summary()
    CFG.validate_required_files()
    stats = compile_train_only_normalization(CFG)
    smoke_test_round_trip(stats, CFG)
