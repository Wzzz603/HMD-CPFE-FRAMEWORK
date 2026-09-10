#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_9_train_mlp_bcc_hcp.py

Dedicated launcher for the Step-10 MLP baseline:

    22 -> 256 -> 256 -> 6
    GELU after each hidden layer
    no recurrence / no history
    dropout = 0 by default

Unlike Linear/Ridge/Lasso classical solvers, MLP is correctly trained with
epochs + Adam/AdamW through the existing 10_7 -> 10_6 neural-network trainer.

The same complete trajectory batches are used as for LSTM, but MLP's Linear
layers operate independently on the final 22-feature dimension, so no
information passes between time points.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class MLPTrainingConfig:
    PHASES: Tuple[str, ...] = ("HCP", "HCP")
    MULTIPLIER: int = 1

    # MLP is a neural-network baseline: use GPU.
    DEVICE: str = "cuda"

    # RAM avoids rereading the large H5 every epoch. Change to "h5" only when
    # system RAM is insufficient.
    DATA_ACCESS_MODE: str = "ram"

    # None -> use BATCH_SIZE from 10_1_config.py.
    BATCH_SIZE_OVERRIDE: int | None = None

    # "never" for a fresh formal run; "auto" to resume an interrupted run.
    RESUME_MODE: str = "never"

    # Reuse compatible Linear train-only normalization when available.
    REUSE_LINEAR_NORMALIZATION: bool = True

    # Normally False.
    NORMALIZATION_OVERWRITE: bool = False

    CHECK_ONLY: bool = False
    STOP_ON_FIRST_FAILURE: bool = True


RUN = MLPTrainingConfig()

_THIS_DIR = Path(__file__).resolve().parent
_TRAIN_ENTRY = _THIS_DIR / "10_7_train.py"


def _import_local(name: str, filename: str):
    path = _THIS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_cfg_mod = _import_local("mlp_launcher_cfg", "10_1_config.py")
_norm_mod = _import_local("mlp_launcher_norm", "10_3_normalizer.py")

BASE_CFG = _cfg_mod.CFG
load_normalization_stats = _norm_mod.load_normalization_stats


def _effective_cfg(phase: str):
    kwargs = {
        "PHASE": str(phase).upper(),
        "MODEL_TYPE": "mlp",
        "MULTIPLIER": int(RUN.MULTIPLIER),
        "DEVICE": str(RUN.DEVICE),
        "DATA_ACCESS_MODE": str(RUN.DATA_ACCESS_MODE),
        "RESUME_MODE": str(RUN.RESUME_MODE),
        "NORMALIZATION_OVERWRITE": bool(RUN.NORMALIZATION_OVERWRITE),
    }
    if RUN.BATCH_SIZE_OVERRIDE is not None:
        kwargs["BATCH_SIZE"] = int(RUN.BATCH_SIZE_OVERRIDE)
    return dataclasses.replace(BASE_CFG, **kwargs)


def _reuse_linear_normalization(phase: str) -> None:
    if (
        not RUN.REUSE_LINEAR_NORMALIZATION
        or RUN.NORMALIZATION_OVERWRITE
    ):
        return

    mlp_cfg = _effective_cfg(phase)
    mlp_path = Path(mlp_cfg.NORMALIZATION_PATH)

    if mlp_path.is_file():
        try:
            load_normalization_stats(mlp_path, mlp_cfg)
        except Exception:
            pass
        else:
            print("[NORMALIZATION] Compatible MLP normalization already exists:")
            print("                ", mlp_path)
            return

    linear_cfg = dataclasses.replace(
        mlp_cfg,
        MODEL_TYPE="linear",
        RESUME_MODE="never",
    )
    linear_path = Path(linear_cfg.NORMALIZATION_PATH)

    if not linear_path.is_file():
        print("[NORMALIZATION] Linear normalization not found; MLP will compile its own.")
        return

    try:
        load_normalization_stats(linear_path, linear_cfg)
    except Exception as exc:
        print("[NORMALIZATION] Linear normalization is incompatible:", exc)
        print("[NORMALIZATION] MLP will compile its own.")
        return

    mlp_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(linear_path, mlp_path)

    src_json = linear_path.with_suffix(".json")
    dst_json = mlp_path.with_suffix(".json")
    if src_json.is_file():
        shutil.copy2(src_json, dst_json)

    # Validate copied file under the MLP effective config.
    load_normalization_stats(mlp_path, mlp_cfg)

    print("[NORMALIZATION] Reused compatible Linear normalization")
    print("        source:", linear_path)
    print("        MLP   :", mlp_path)


def build_command(phase: str) -> list[str]:
    command = [
        sys.executable,
        str(_TRAIN_ENTRY),
        "--phase", str(phase).upper(),
        "--model", "mlp",
        "--multiplier", str(int(RUN.MULTIPLIER)),
        "--device", str(RUN.DEVICE),
        "--data-access", str(RUN.DATA_ACCESS_MODE),
        "--resume", str(RUN.RESUME_MODE),
        "--no-window",
    ]

    if RUN.BATCH_SIZE_OVERRIDE is not None:
        command.extend([
            "--batch-size",
            str(int(RUN.BATCH_SIZE_OVERRIDE)),
        ])

    if RUN.NORMALIZATION_OVERWRITE:
        command.append("--normalization-overwrite")

    if RUN.CHECK_ONLY:
        command.append("--check-only")

    return command


def run_phase(phase: str) -> int:
    phase = str(phase).upper().strip()
    _reuse_linear_normalization(phase)

    cfg = _effective_cfg(phase)
    command = build_command(phase)

    print("=" * 108)
    print("Step-10 MLP training")
    print("Phase             :", phase)
    print("Architecture      : 22 -> 256 -> 256 -> 6")
    print("Activation        :", cfg.MLP_ACTIVATION)
    print("Dropout           :", cfg.MLP_DROPOUT)
    print("MLP weight decay  :", cfg.MLP_WEIGHT_DECAY)
    print("Multiplier        :", cfg.MULTIPLIER)
    print("Batch size        :", cfg.BATCH_SIZE)
    print("Device            :", cfg.DEVICE)
    print("Data access       :", cfg.DATA_ACCESS_MODE)
    print("Resume            :", cfg.RESUME_MODE)
    print("Uses history      : NO")
    print("Training epochs   : YES (correct for MLP)")
    print("Experiment root   :", cfg.EXPERIMENT_ROOT)
    print("Command           :")
    print(" ".join(f'"{x}"' if " " in x else x for x in command))
    print("=" * 108)

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(_THIS_DIR),
        check=False,
    )
    elapsed = time.perf_counter() - started

    print("-" * 108)
    print(f"{phase} return code : {completed.returncode}")
    print(f"{phase} elapsed     : {elapsed:.2f} s")
    print("-" * 108)

    return int(completed.returncode)


def main() -> int:
    results = {}

    for phase in RUN.PHASES:
        phase = str(phase).upper().strip()
        code = run_phase(phase)
        results[phase] = code

        if code != 0 and RUN.STOP_ON_FIRST_FAILURE:
            break

    print("=" * 108)
    print("MLP launch summary")
    for phase in RUN.PHASES:
        phase = str(phase).upper().strip()
        if phase not in results:
            status = "NOT RUN"
        else:
            status = "PASS" if results[phase] == 0 else f"FAILED({results[phase]})"
        print(f"{phase:>4s} : {status}")
    print("=" * 108)

    return 0 if results and all(v == 0 for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
