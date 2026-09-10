#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_7_train.py

Formal Step-10 training entry.

Examples
--------
Use the selections written in 10_1_config.py:

    python 10_7_train.py

Override only this launch without editing the config file:

    python 10_7_train.py --phase HCP --model lstm_fc --multiplier 3
    python 10_7_train.py --phase HCP --model ridge --multiplier 3
    python 10_7_train.py --resume required --max-epochs 500
    python 10_7_train.py --check-only

Resume behavior
---------------
The default resume mode is "auto":

- no last_checkpoint.pt:
      begin at epoch 1;
- last_checkpoint.pt exists:
      restore the full original history, graph, optimizer, scheduler, AMP,
      cumulative time, LCS crossing state and random states, then continue.

The command-line overrides are converted into a fresh Config object, so all
derived H5/checkpoint/output paths are recalculated consistently.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import torch


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
    "step10_config_for_entry",
    "10_1_config.py",
)
_dataset_module = _import_local_module(
    "step10_dataset_for_entry",
    "10_2_h5_dataset.py",
)
_normalizer_module = _import_local_module(
    "step10_normalizer_for_entry",
    "10_3_normalizer.py",
)
_models_module = _import_local_module(
    "step10_models_for_entry",
    "10_4_models.py",
)
_trainer_module = _import_local_module(
    "step10_trainer_for_entry",
    "10_6_trainer.py",
)

BASE_CFG = _config_module.CFG
Config = _config_module.Config
ALLOWED_PHASES = tuple(_config_module.ALLOWED_PHASES)
ALLOWED_MODEL_TYPES = tuple(_config_module.ALLOWED_MODEL_TYPES)

set_global_seed = _config_module.set_global_seed
inspect_phase_h5 = _dataset_module.inspect_phase_h5
load_lcs_reference = _dataset_module.load_lcs_reference
load_normalization_stats = _normalizer_module.load_normalization_stats
build_model = _models_module.build_model
get_model_info = _models_module.get_model_info
smoke_test_model = _models_module.smoke_test_model
train_selected_model = _trainer_module.train_selected_model


# =============================================================================
# 1. CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one Step-10 CPFE stress-initialization model. "
            "Unspecified options are taken from 10_1_config.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--phase",
        choices=ALLOWED_PHASES,
        default=None,
        help="Phase model to train.",
    )
    parser.add_argument(
        "--model",
        choices=ALLOWED_MODEL_TYPES,
        default=None,
        help="Model type.",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=None,
        help="Increment multiplier.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help=(
            "Final epoch number. During resume, this is the total target epoch, "
            "not the number of additional epochs."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Complete trajectories per training batch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device, for example cuda, cuda:0 or cpu.",
    )
    parser.add_argument(
        "--data-access",
        choices=("ram", "h5"),
        default=None,
        help="Core Step-08 H5 access mode.",
    )
    parser.add_argument(
        "--resume",
        choices=("auto", "never", "required"),
        default=None,
        help="Checkpoint resume policy.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional explicit checkpoint path. The checkpoint still must match "
            "the selected phase/multiplier/model."
        ),
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help=(
            "Do not open an interactive Matplotlib window. The complete live "
            "PNG is still updated."
        ),
    )
    parser.add_argument(
        "--normalization-overwrite",
        action="store_true",
        help="Recompute the train-only normalization file.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Validate H5/LCS/model/config and exit without starting training."
        ),
    )

    return parser


def build_effective_config(args: argparse.Namespace) -> Config:
    overrides: Dict[str, object] = {}

    if args.phase is not None:
        overrides["PHASE"] = args.phase
    if args.model is not None:
        overrides["MODEL_TYPE"] = args.model
    if args.multiplier is not None:
        overrides["MULTIPLIER"] = int(args.multiplier)
    if args.max_epochs is not None:
        overrides["MAX_EPOCHS"] = int(args.max_epochs)
    if args.batch_size is not None:
        overrides["BATCH_SIZE"] = int(args.batch_size)
    if args.device is not None:
        overrides["DEVICE"] = str(args.device)
    if args.data_access is not None:
        overrides["DATA_ACCESS_MODE"] = args.data_access
    if args.resume is not None:
        overrides["RESUME_MODE"] = args.resume
    if args.resume_checkpoint is not None:
        overrides["RESUME_CHECKPOINT_PATH"] = str(
            args.resume_checkpoint
        )
    if args.no_window:
        overrides["DASHBOARD_SHOW_WINDOW"] = False
    if args.normalization_overwrite:
        overrides["NORMALIZATION_OVERWRITE"] = True

    # dataclasses.replace invokes Config.__init__ and __post_init__, so every
    # derived path is recalculated after phase/model/multiplier overrides.
    cfg = dataclasses.replace(BASE_CFG, **overrides)
    cfg.validate()
    return cfg


# =============================================================================
# 2. Environment and preflight
# =============================================================================

def _now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def collect_environment() -> Dict[str, object]:
    cuda_available = bool(torch.cuda.is_available())

    environment: Dict[str, object] = {
        "time": _now_string(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "working_directory": os.getcwd(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "h5py_version": h5py.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if cuda_available
            else None
        ),
        "cuda_device_count": (
            int(torch.cuda.device_count())
            if cuda_available
            else 0
        ),
    }

    if cuda_available:
        environment["cuda_devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gib": (
                    torch.cuda.get_device_properties(i).total_memory
                    / float(1024 ** 3)
                ),
            }
            for i in range(torch.cuda.device_count())
        ]

    return environment


def run_preflight(cfg: Config) -> Dict[str, object]:
    cfg.validate_required_files()

    info = inspect_phase_h5(cfg)
    lcs = load_lcs_reference(cfg)

    if cfg.NUM_WORKERS != 0:
        print(
            "[PREFLIGHT WARNING] Exact epoch-boundary shuffle continuation is "
            "most reliable with NUM_WORKERS=0. Current value=%d."
            % cfg.NUM_WORKERS
        )

    if cfg.DATA_ACCESS_MODE == "ram":
        print(
            "[PREFLIGHT] The core arrays will occupy approximately %.3f GiB "
            "in system RAM."
            % info.estimated_core_gib
        )

    normalization_exists = Path(cfg.NORMALIZATION_PATH).is_file()
    normalization_status = "absent"

    normalization_reason = None

    if normalization_exists and not cfg.NORMALIZATION_OVERWRITE:
        try:
            stats = load_normalization_stats(
                Path(cfg.NORMALIZATION_PATH),
                cfg,
            )
        except Exception as exc:
            normalization_status = "incompatible_will_recompute"
            normalization_reason = str(exc)
            print(
                "[PREFLIGHT WARNING] Existing normalization file is "
                "incompatible and will be recomputed before training: %s"
                % exc
            )
        else:
            normalization_status = "compatible"
            del stats
    elif cfg.NORMALIZATION_OVERWRITE:
        normalization_status = "will_recompute"
    else:
        normalization_status = "will_create"

    model = build_model(cfg)
    smoke_test_model(model, cfg)
    model_info = get_model_info(model, cfg)
    del model

    checkpoint_path = (
        Path(cfg.RESUME_CHECKPOINT_PATH)
        if str(cfg.RESUME_CHECKPOINT_PATH).strip()
        else Path(cfg.LAST_CHECKPOINT_PATH)
    )

    checkpoint_exists = checkpoint_path.is_file()

    print("=" * 108)
    print("Step-10 preflight: PASS")
    print("Phase                :", info.phase)
    print("Multiplier           :", info.multiplier)
    print("Sequence length      :", info.sequence_length)
    print("Train trajectories   :", info.train_count)
    print("Val trajectories     :", info.val_count)
    print("Model                :", model_info.display_name)
    print("Trainable parameters :", model_info.parameter_count_trainable)
    print("Uses sequence history:", model_info.uses_sequence_history)
    print("LCS validation line  : %.6f %%" % lcs.horizontal_line_pct)
    print("Normalization        :", normalization_status)
    print("Resume mode          :", cfg.RESUME_MODE)
    print("Checkpoint exists    :", checkpoint_exists)
    print("Checkpoint path      :", checkpoint_path)
    print("History CSV          :", cfg.HISTORY_CSV_PATH)
    print("Live dashboard       :", cfg.LIVE_DASHBOARD_PATH)
    print("=" * 108)

    return {
        "phase_h5": str(info.path),
        "phase": info.phase,
        "multiplier": info.multiplier,
        "sequence_length": info.sequence_length,
        "trajectory_count": info.trajectory_count,
        "train_count": info.train_count,
        "val_count": info.val_count,
        "estimated_core_gib": info.estimated_core_gib,
        "model": model_info.to_dict(),
        "lcs_line_pct": lcs.horizontal_line_pct,
        "normalization_status": normalization_status,
        "normalization_reason": normalization_reason,
        "normalization_path": cfg.NORMALIZATION_PATH,
        "resume_mode": cfg.RESUME_MODE,
        "resume_checkpoint": str(checkpoint_path),
        "checkpoint_exists": checkpoint_exists,
    }


def write_launch_manifest(
    cfg: Config,
    environment: Dict[str, object],
    preflight: Dict[str, object],
    argv: list[str],
) -> Path:
    cfg.create_output_directories()

    path = (
        Path(cfg.EXPERIMENT_ROOT)
        / "launch_manifest.json"
    )

    payload = {
        "schema_name": "CPFE_STEP10_LAUNCH_MANIFEST_V1",
        "launch_time": _now_string(),
        "command_line": argv,
        "environment": environment,
        "preflight": preflight,
        "effective_config": cfg.snapshot_dict(),
    }

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)

    return path


# =============================================================================
# 3. Main
# =============================================================================

def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = build_effective_config(args)
    set_global_seed(cfg)

    environment = collect_environment()

    print("=" * 108)
    print("Step-10 formal training entry")
    print("Launch time     :", environment["time"])
    print("Python          :", environment["python"].split()[0])
    print("PyTorch         :", environment["torch_version"])
    print("CUDA available  :", environment["cuda_available"])
    print("Selected device :", cfg.DEVICE)
    print("=" * 108)

    preflight = run_preflight(cfg)

    manifest = write_launch_manifest(
        cfg=cfg,
        environment=environment,
        preflight=preflight,
        argv=(
            [sys.executable, str(Path(__file__).resolve())]
            + list(sys.argv[1:])
        ),
    )
    print("Launch manifest :", manifest)

    if args.check_only:
        print("Check-only mode: no training was started.")
        return 0

    summary = train_selected_model(cfg)

    stopped_reason = str(
        summary.get("stopped_reason", "")
    )
    if stopped_reason == "exception":
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
