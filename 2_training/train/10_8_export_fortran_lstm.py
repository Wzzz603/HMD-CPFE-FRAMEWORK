#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10_8_export_fortran_lstm.py

Export the trained Step-10 PyTorch LSTM checkpoint to the plain-text format
consumed by the existing Abaqus/Fortran READ_SINGLE_LSTM() routine.

Expected Fortran reader layout
------------------------------
1) Header:
       N_in, N_hidden, N_layers, N_out

2) Normalization:
       input_mean[22]
       input_std[22]
       output_mean[6]
       output_std[6]

3) For each LSTM layer:
       weight_ih [4H, input_dim_of_this_layer]
       weight_hh [4H, H]
       bias_ih   [4H]
       bias_hh   [4H]

4) FC head:
       weight_fc [N_out, H]
       bias_fc   [N_out]

IMPORTANT
---------
Fortran list-directed READ fills rank-2 arrays in Fortran column-major order.
Therefore every 2-D matrix is exported using NumPy flatten(order="F").

The exporter also performs two verification stages:
A) write -> read-back exact tensor comparison;
B) PyTorch forward -> Python emulation of the Fortran LSTM forward pass.

This script supports the current Fortran deployment interface:
    input=22, hidden=dynamic, layers=dynamic, output=6, unidirectional,
    nn.LSTM + one nn.Linear output head.

Hidden size and number of LSTM layers are inferred directly from the checkpoint.
It will FAIL loudly if the checkpoint architecture is incompatible with the
current mlmodule.f assumptions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch


# =============================================================================
# 0. User configuration
# =============================================================================

@dataclass
class ExportConfig:
    # Directory containing:
    #   checkpoints/best_checkpoint.pt
    #   normalization_train_only.npz
    #
    # Example:
    # F:\...\10_training\05_simple_external27__BCC__m03__lstm_fc__Uni2xLSTM256_FC6
    EXPERIMENT_ROOT: str = (
        r"F:\1_job\20251124\CPFE_PARAMATER_TEST_PBC\2_RVE_BUILD"
        r"\10_training"
        r"\05_simple_external27__HCP__m01__lstm_fc__Uni2xLSTM256_FC6"
    )

    # "auto" uses phase stored inside checkpoint/config and creates:
    #   BCC_best.txt or HCP_best.txt
    OUTPUT_NAME: str = "auto"

    # Use the best checkpoint for Abaqus deployment.
    CHECKPOINT_RELATIVE_PATH: str = r"checkpoints\best_checkpoint.pt"
    NORMALIZATION_RELATIVE_PATH: str = "normalization_train_only.npz"
    EXPORT_SUBDIR: str = "exports"

    # Numerical format used in the txt.
    FLOAT_FORMAT: str = ".17e"
    VALUES_PER_LINE: int = 8

    # Verification controls.
    VERIFY_READBACK: bool = True
    VERIFY_FORWARD: bool = True
    VERIFY_SEQUENCE_LENGTH: int = 9
    VERIFY_BATCH_SIZE: int = 4
    VERIFY_RANDOM_SEED: int = 20260727

    # Tolerances. Text is written as float64 precision, while PyTorch model
    # normally operates in float32, so forward agreement is judged accordingly.
    READBACK_ATOL: float = 1.0e-13
    READBACK_RTOL: float = 1.0e-13
    FORWARD_ATOL: float = 2.0e-5
    FORWARD_RTOL: float = 2.0e-5

    # Strict compatibility with the current mlmodule.f interface.
    # Hidden size and number of layers are dynamic and are read from checkpoint.
    EXPECTED_INPUT_DIM: int = 22
    EXPECTED_OUTPUT_DIM: int = 6


CFG = ExportConfig()


# =============================================================================
# 1. Utilities
# =============================================================================

def _load_checkpoint(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not expose weights_only.
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Checkpoint must be a mapping/dict, got {type(checkpoint).__name__}"
        )
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'. "
            "This exporter targets the current 10_6_trainer.py checkpoint schema."
        )
    return checkpoint


def _tensor_to_numpy(
    state: Mapping[str, torch.Tensor],
    key: str,
) -> np.ndarray:
    if key not in state:
        raise KeyError(f"Missing model parameter: {key}")
    tensor = state[key]
    if not torch.is_tensor(tensor):
        raise TypeError(f"{key} is not a torch.Tensor")
    return (
        tensor.detach()
        .cpu()
        .to(dtype=torch.float64)
        .numpy()
        .copy()
    )


def _decode_scalar_string(value) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


@dataclass(frozen=True)
class NormalizationData:
    input_mean: np.ndarray
    input_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray
    phase: str | None
    multiplier: int | None
    sequence_length: int | None
    input_names: Tuple[str, ...]
    target_names: Tuple[str, ...]


def _load_normalization(path: Path) -> NormalizationData:
    if not path.is_file():
        raise FileNotFoundError(f"Normalization file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = (
            "input_mean",
            "input_std",
            "target_mean",
            "target_std",
        )
        for name in required:
            if name not in data:
                raise KeyError(
                    f"Missing {name!r} in normalization file: {path}"
                )

        input_mean = np.asarray(data["input_mean"], dtype=np.float64).reshape(-1)
        input_std = np.asarray(data["input_std"], dtype=np.float64).reshape(-1)
        output_mean = np.asarray(data["target_mean"], dtype=np.float64).reshape(-1)
        output_std = np.asarray(data["target_std"], dtype=np.float64).reshape(-1)

        input_names: Tuple[str, ...] = ()
        target_names: Tuple[str, ...] = ()
        if "input_names" in data:
            input_names = tuple(
                _decode_scalar_string(x)
                for x in np.asarray(data["input_names"]).reshape(-1)
            )
        if "target_names" in data:
            target_names = tuple(
                _decode_scalar_string(x)
                for x in np.asarray(data["target_names"]).reshape(-1)
            )

        phase = None
        multiplier = None
        sequence_length = None

        if "metadata_json" in data:
            metadata_text = _decode_scalar_string(data["metadata_json"])
            metadata = json.loads(metadata_text)
            raw_phase = metadata.get("phase")
            if raw_phase is not None:
                phase = str(raw_phase).upper().strip()
            raw_multiplier = metadata.get("multiplier")
            if raw_multiplier is not None:
                multiplier = int(raw_multiplier)
            raw_seq = metadata.get("sequence_length")
            if raw_seq is not None:
                sequence_length = int(raw_seq)

    if not np.isfinite(input_mean).all():
        raise ValueError("input_mean contains NaN/Inf")
    if not np.isfinite(input_std).all():
        raise ValueError("input_std contains NaN/Inf")
    if not np.isfinite(output_mean).all():
        raise ValueError("target_mean contains NaN/Inf")
    if not np.isfinite(output_std).all():
        raise ValueError("target_std contains NaN/Inf")
    if np.any(input_std <= 0.0):
        raise ValueError("input_std must be strictly positive")
    if np.any(output_std <= 0.0):
        raise ValueError("target_std must be strictly positive")

    return NormalizationData(
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
        phase=phase,
        multiplier=multiplier,
        sequence_length=sequence_length,
        input_names=input_names,
        target_names=target_names,
    )


@dataclass(frozen=True)
class LSTMParameters:
    n_in: int
    n_hidden: int
    n_layers: int
    n_out: int
    weight_ih: Tuple[np.ndarray, ...]
    weight_hh: Tuple[np.ndarray, ...]
    bias_ih: Tuple[np.ndarray, ...]
    bias_hh: Tuple[np.ndarray, ...]
    weight_fc: np.ndarray
    bias_fc: np.ndarray


def _infer_parameters(
    state: Mapping[str, torch.Tensor],
) -> LSTMParameters:
    # Current model parameter names:
    #   lstm.weight_ih_l0
    #   lstm.weight_hh_l0
    #   lstm.bias_ih_l0
    #   lstm.bias_hh_l0
    #   ...
    #   fc_head.weight
    #   fc_head.bias

    layer_indices: List[int] = []
    prefix = "lstm.weight_ih_l"
    for key in state.keys():
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            if suffix.isdigit():
                layer_indices.append(int(suffix))

    layer_indices = sorted(set(layer_indices))
    if not layer_indices:
        raise KeyError(
            "No parameters named 'lstm.weight_ih_l*' were found."
        )
    expected = list(range(max(layer_indices) + 1))
    if layer_indices != expected:
        raise ValueError(
            f"LSTM layer indices are not contiguous: {layer_indices}"
        )

    n_layers = len(layer_indices)

    wih_list: List[np.ndarray] = []
    whh_list: List[np.ndarray] = []
    bih_list: List[np.ndarray] = []
    bhh_list: List[np.ndarray] = []

    n_hidden = None
    n_in = None

    for layer in layer_indices:
        # Reject bidirectional checkpoints explicitly.
        reverse_keys = [
            f"lstm.weight_ih_l{layer}_reverse",
            f"lstm.weight_hh_l{layer}_reverse",
            f"lstm.bias_ih_l{layer}_reverse",
            f"lstm.bias_hh_l{layer}_reverse",
        ]
        if any(key in state for key in reverse_keys):
            raise ValueError(
                "Bidirectional LSTM checkpoint detected. "
                "Current mlmodule.f supports only unidirectional LSTM."
            )

        wih = _tensor_to_numpy(state, f"lstm.weight_ih_l{layer}")
        whh = _tensor_to_numpy(state, f"lstm.weight_hh_l{layer}")
        bih = _tensor_to_numpy(state, f"lstm.bias_ih_l{layer}").reshape(-1)
        bhh = _tensor_to_numpy(state, f"lstm.bias_hh_l{layer}").reshape(-1)

        if wih.ndim != 2 or whh.ndim != 2:
            raise ValueError("LSTM weights must be rank-2 arrays")
        if wih.shape[0] % 4 != 0:
            raise ValueError(
                f"Invalid LSTM gate dimension: {wih.shape}"
            )

        this_hidden = wih.shape[0] // 4
        if n_hidden is None:
            n_hidden = this_hidden
        elif this_hidden != n_hidden:
            raise ValueError("Different hidden sizes across LSTM layers")

        if whh.shape != (4 * n_hidden, n_hidden):
            raise ValueError(
                f"Invalid weight_hh_l{layer} shape {whh.shape}; "
                f"expected {(4*n_hidden, n_hidden)}"
            )
        if bih.shape != (4 * n_hidden,):
            raise ValueError(
                f"Invalid bias_ih_l{layer} shape {bih.shape}"
            )
        if bhh.shape != (4 * n_hidden,):
            raise ValueError(
                f"Invalid bias_hh_l{layer} shape {bhh.shape}"
            )

        if layer == 0:
            n_in = int(wih.shape[1])
        else:
            if wih.shape[1] != n_hidden:
                raise ValueError(
                    f"Layer {layer} input size={wih.shape[1]}, "
                    f"expected hidden size={n_hidden}"
                )

        wih_list.append(wih)
        whh_list.append(whh)
        bih_list.append(bih)
        bhh_list.append(bhh)

    if n_in is None or n_hidden is None:
        raise RuntimeError("Failed to infer LSTM dimensions")

    weight_fc = _tensor_to_numpy(state, "fc_head.weight")
    bias_fc = _tensor_to_numpy(state, "fc_head.bias").reshape(-1)

    if weight_fc.ndim != 2:
        raise ValueError("fc_head.weight must be rank 2")

    n_out = int(weight_fc.shape[0])
    if weight_fc.shape[1] != n_hidden:
        raise ValueError(
            f"fc_head.weight shape={weight_fc.shape}; "
            f"expected ({n_out}, {n_hidden})"
        )
    if bias_fc.shape != (n_out,):
        raise ValueError(
            f"fc_head.bias shape={bias_fc.shape}; expected {(n_out,)}"
        )

    return LSTMParameters(
        n_in=int(n_in),
        n_hidden=int(n_hidden),
        n_layers=int(n_layers),
        n_out=int(n_out),
        weight_ih=tuple(wih_list),
        weight_hh=tuple(whh_list),
        bias_ih=tuple(bih_list),
        bias_hh=tuple(bhh_list),
        weight_fc=weight_fc,
        bias_fc=bias_fc,
    )


def _validate_fortran_compatibility(
    params: LSTMParameters,
    norm: NormalizationData,
    checkpoint: Mapping[str, object],
    cfg: ExportConfig,
) -> str:
    problems: List[str] = []

    if params.n_in != cfg.EXPECTED_INPUT_DIM:
        problems.append(
            f"input dim={params.n_in}, Fortran expects {cfg.EXPECTED_INPUT_DIM}"
        )
    if params.n_out != cfg.EXPECTED_OUTPUT_DIM:
        problems.append(
            f"output dim={params.n_out}, Fortran expects {cfg.EXPECTED_OUTPUT_DIM}"
        )

    if norm.input_mean.shape != (params.n_in,):
        problems.append(
            f"normalization input_mean shape={norm.input_mean.shape}, "
            f"expected {(params.n_in,)}"
        )
    if norm.input_std.shape != (params.n_in,):
        problems.append(
            f"normalization input_std shape={norm.input_std.shape}, "
            f"expected {(params.n_in,)}"
        )
    if norm.output_mean.shape != (params.n_out,):
        problems.append(
            f"normalization target_mean shape={norm.output_mean.shape}, "
            f"expected {(params.n_out,)}"
        )
    if norm.output_std.shape != (params.n_out,):
        problems.append(
            f"normalization target_std shape={norm.output_std.shape}, "
            f"expected {(params.n_out,)}"
        )

    phase_ckpt = checkpoint.get("phase")
    phase_ckpt = (
        str(phase_ckpt).upper().strip()
        if phase_ckpt is not None
        else None
    )

    if phase_ckpt is not None and norm.phase is not None:
        if phase_ckpt != norm.phase:
            problems.append(
                f"checkpoint phase={phase_ckpt}, normalization phase={norm.phase}"
            )

    multiplier_ckpt = checkpoint.get("multiplier")
    if multiplier_ckpt is not None and norm.multiplier is not None:
        if int(multiplier_ckpt) != int(norm.multiplier):
            problems.append(
                f"checkpoint multiplier={multiplier_ckpt}, "
                f"normalization multiplier={norm.multiplier}"
            )

    model_type = checkpoint.get("model_type")
    if model_type is not None and str(model_type).lower().strip() != "lstm_fc":
        problems.append(
            f"checkpoint model_type={model_type!r}; only lstm_fc can be exported"
        )

    # Current Fortran assumes exact PyTorch LSTM gate ordering i,f,g,o.
    # PyTorch uses the same gate order.
    if problems:
        raise RuntimeError(
            "Checkpoint is NOT compatible with the current mlmodule.f:\n- "
            + "\n- ".join(problems)
        )

    phase = phase_ckpt or norm.phase
    if phase not in ("BCC", "HCP"):
        raise RuntimeError(
            "Could not determine phase as BCC or HCP from checkpoint/"
            "normalization metadata."
        )
    return phase


def _iter_formatted(
    values: np.ndarray,
    fmt: str,
    per_line: int,
):
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    for start in range(0, flat.size, per_line):
        part = flat[start:start + per_line]
        yield " ".join(format(float(x), fmt) for x in part)


def _write_array(
    f,
    array: np.ndarray,
    cfg: ExportConfig,
    *,
    fortran_matrix_order: bool,
) -> None:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 2 and fortran_matrix_order:
        values = arr.flatten(order="F")
    else:
        values = arr.reshape(-1)

    for line in _iter_formatted(
        values,
        cfg.FLOAT_FORMAT,
        int(cfg.VALUES_PER_LINE),
    ):
        f.write(line + "\n")


def _export_txt(
    output_path: Path,
    params: LSTMParameters,
    norm: NormalizationData,
    cfg: ExportConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="ascii",
        newline="\n",
    ) as f:
        # Header.
        f.write(
            f"{params.n_in} {params.n_hidden} "
            f"{params.n_layers} {params.n_out}\n"
        )

        # One Fortran READ statement consumes these arrays sequentially.
        _write_array(
            f, norm.input_mean, cfg,
            fortran_matrix_order=False,
        )
        _write_array(
            f, norm.input_std, cfg,
            fortran_matrix_order=False,
        )
        _write_array(
            f, norm.output_mean, cfg,
            fortran_matrix_order=False,
        )
        _write_array(
            f, norm.output_std, cfg,
            fortran_matrix_order=False,
        )

        # LSTM layers.
        for layer in range(params.n_layers):
            _write_array(
                f, params.weight_ih[layer], cfg,
                fortran_matrix_order=True,
            )
            _write_array(
                f, params.weight_hh[layer], cfg,
                fortran_matrix_order=True,
            )
            _write_array(
                f, params.bias_ih[layer], cfg,
                fortran_matrix_order=False,
            )
            _write_array(
                f, params.bias_hh[layer], cfg,
                fortran_matrix_order=False,
            )

        # FC.
        _write_array(
            f, params.weight_fc, cfg,
            fortran_matrix_order=True,
        )
        _write_array(
            f, params.bias_fc, cfg,
            fortran_matrix_order=False,
        )

    os.replace(temp_path, output_path)


# =============================================================================
# 2. Read-back parser that emulates Fortran list-directed numeric consumption
# =============================================================================

@dataclass(frozen=True)
class ExportedText:
    params: LSTMParameters
    norm: NormalizationData


def _read_all_numeric_tokens(path: Path) -> List[str]:
    text = path.read_text(encoding="ascii")
    # Exporter writes whitespace-separated numeric tokens only.
    return text.split()


def _consume(
    tokens: List[str],
    pos: int,
    count: int,
    dtype=float,
):
    end = pos + count
    if end > len(tokens):
        raise EOFError(
            f"Unexpected EOF: need {count} tokens at position {pos}, "
            f"only {len(tokens)-pos} remain"
        )
    values = [dtype(tokens[i]) for i in range(pos, end)]
    return values, end


def _read_exported_txt(path: Path) -> ExportedText:
    tokens = _read_all_numeric_tokens(path)
    pos = 0

    header, pos = _consume(tokens, pos, 4, int)
    n_in, n_hidden, n_layers, n_out = header

    def read_vector(n: int) -> np.ndarray:
        nonlocal pos
        vals, pos = _consume(tokens, pos, n, float)
        return np.asarray(vals, dtype=np.float64)

    def read_matrix(shape: Tuple[int, int]) -> np.ndarray:
        nonlocal pos
        count = int(shape[0] * shape[1])
        vals, pos = _consume(tokens, pos, count, float)
        # Fortran READ filled this matrix in column-major element order.
        return np.asarray(
            vals,
            dtype=np.float64,
        ).reshape(shape, order="F")

    input_mean = read_vector(n_in)
    input_std = read_vector(n_in)
    output_mean = read_vector(n_out)
    output_std = read_vector(n_out)

    wih: List[np.ndarray] = []
    whh: List[np.ndarray] = []
    bih: List[np.ndarray] = []
    bhh: List[np.ndarray] = []

    for layer in range(n_layers):
        cur_in = n_in if layer == 0 else n_hidden
        wih.append(read_matrix((4 * n_hidden, cur_in)))
        whh.append(read_matrix((4 * n_hidden, n_hidden)))
        bih.append(read_vector(4 * n_hidden))
        bhh.append(read_vector(4 * n_hidden))

    weight_fc = read_matrix((n_out, n_hidden))
    bias_fc = read_vector(n_out)

    if pos != len(tokens):
        raise ValueError(
            f"Unexpected extra numeric tokens in export: "
            f"consumed {pos}, total {len(tokens)}"
        )

    return ExportedText(
        params=LSTMParameters(
            n_in=n_in,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_out=n_out,
            weight_ih=tuple(wih),
            weight_hh=tuple(whh),
            bias_ih=tuple(bih),
            bias_hh=tuple(bhh),
            weight_fc=weight_fc,
            bias_fc=bias_fc,
        ),
        norm=NormalizationData(
            input_mean=input_mean,
            input_std=input_std,
            output_mean=output_mean,
            output_std=output_std,
            phase=None,
            multiplier=None,
            sequence_length=None,
            input_names=(),
            target_names=(),
        ),
    )


def _assert_close(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    cfg: ExportConfig,
) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape {actual.shape} != {expected.shape}"
        )

    if not np.allclose(
        actual,
        expected,
        atol=cfg.READBACK_ATOL,
        rtol=cfg.READBACK_RTOL,
    ):
        diff = np.abs(actual - expected)
        idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
        raise AssertionError(
            f"{name}: read-back mismatch; max_abs={diff[idx]:.6e} "
            f"at index={idx}, read={actual[idx]:.17e}, "
            f"expected={expected[idx]:.17e}"
        )


def _verify_readback(
    exported: ExportedText,
    params: LSTMParameters,
    norm: NormalizationData,
    cfg: ExportConfig,
) -> None:
    p = exported.params
    if (
        p.n_in, p.n_hidden, p.n_layers, p.n_out
    ) != (
        params.n_in, params.n_hidden,
        params.n_layers, params.n_out,
    ):
        raise AssertionError("Header read-back mismatch")

    _assert_close(
        "input_mean",
        exported.norm.input_mean,
        norm.input_mean,
        cfg,
    )
    _assert_close(
        "input_std",
        exported.norm.input_std,
        norm.input_std,
        cfg,
    )
    _assert_close(
        "output_mean",
        exported.norm.output_mean,
        norm.output_mean,
        cfg,
    )
    _assert_close(
        "output_std",
        exported.norm.output_std,
        norm.output_std,
        cfg,
    )

    for layer in range(params.n_layers):
        _assert_close(
            f"weight_ih_l{layer}",
            p.weight_ih[layer],
            params.weight_ih[layer],
            cfg,
        )
        _assert_close(
            f"weight_hh_l{layer}",
            p.weight_hh[layer],
            params.weight_hh[layer],
            cfg,
        )
        _assert_close(
            f"bias_ih_l{layer}",
            p.bias_ih[layer],
            params.bias_ih[layer],
            cfg,
        )
        _assert_close(
            f"bias_hh_l{layer}",
            p.bias_hh[layer],
            params.bias_hh[layer],
            cfg,
        )

    _assert_close(
        "fc_head.weight",
        p.weight_fc,
        params.weight_fc,
        cfg,
    )
    _assert_close(
        "fc_head.bias",
        p.bias_fc,
        params.bias_fc,
        cfg,
    )


# =============================================================================
# 3. Forward verification
# =============================================================================

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid implementation.
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0.0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    return out


def _fortran_style_forward_sequence(
    x_physical: np.ndarray,
    exported: ExportedText,
) -> np.ndarray:
    """
    Emulate mlmodule.f's LSTM math up to the crystal-frame FC output and
    denormalization.

    NOTE:
    The uploaded mlmodule.f subsequently rotates the predicted stress using
    theta entries 14:22. That rotation is outside the PyTorch model itself.
    Therefore this verification compares the neural-network output BEFORE the
    Fortran post-rotation step, which is exactly what must match PyTorch.
    """
    p = exported.params
    n = exported.norm

    x = np.asarray(x_physical, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != p.n_in:
        raise ValueError(
            f"x_physical must be [B,L,{p.n_in}], got {x.shape}"
        )

    batch, length, _ = x.shape
    y = np.zeros((batch, length, p.n_out), dtype=np.float64)

    # PyTorch defaults to zero h/c at the start of each sequence.
    h = np.zeros(
        (p.n_layers, batch, p.n_hidden),
        dtype=np.float64,
    )
    c = np.zeros_like(h)

    x_norm = (x - n.input_mean[None, None, :]) / n.input_std[None, None, :]

    for t in range(length):
        current = x_norm[:, t, :]

        for layer in range(p.n_layers):
            gates = (
                current @ p.weight_ih[layer].T
                + h[layer] @ p.weight_hh[layer].T
                + p.bias_ih[layer][None, :]
                + p.bias_hh[layer][None, :]
            )

            H = p.n_hidden
            i_gate = _sigmoid_np(gates[:, 0:H])
            f_gate = _sigmoid_np(gates[:, H:2*H])
            g_gate = np.tanh(gates[:, 2*H:3*H])
            o_gate = _sigmoid_np(gates[:, 3*H:4*H])

            c[layer] = (
                f_gate * c[layer]
                + i_gate * g_gate
            )
            h[layer] = (
                o_gate * np.tanh(c[layer])
            )

            current = h[layer]

        pred_norm = (
            current @ p.weight_fc.T
            + p.bias_fc[None, :]
        )
        y[:, t, :] = (
            pred_norm * n.output_std[None, :]
            + n.output_mean[None, :]
        )

    return y


def _build_torch_reference_from_checkpoint(
    checkpoint: Mapping[str, object],
    params: LSTMParameters,
) -> Tuple[torch.nn.LSTM, torch.nn.Linear]:
    """
    Build a minimal PyTorch reference network directly from checkpoint shapes.

    This deliberately does NOT import 10_4_models.py or read 10_1_config.py.
    Therefore hidden size and number of layers may differ between experiments
    (for example BCC=128x2 and HCP=256x2) without editing the exporter.
    """
    raw_state = checkpoint.get("model_state_dict")
    if not isinstance(raw_state, Mapping):
        raise TypeError("model_state_dict is not a mapping")

    state: Mapping[str, torch.Tensor] = raw_state  # type: ignore[assignment]

    lstm = torch.nn.LSTM(
        input_size=params.n_in,
        hidden_size=params.n_hidden,
        num_layers=params.n_layers,
        bias=True,
        batch_first=True,
        dropout=0.0,
        bidirectional=False,
    )
    fc = torch.nn.Linear(
        in_features=params.n_hidden,
        out_features=params.n_out,
        bias=True,
    )

    lstm_state: Dict[str, torch.Tensor] = {}
    for layer in range(params.n_layers):
        for stem in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            source_key = f"lstm.{stem}_l{layer}"
            target_key = f"{stem}_l{layer}"
            if source_key not in state:
                raise KeyError(f"Missing model parameter: {source_key}")
            tensor = state[source_key]
            if not torch.is_tensor(tensor):
                raise TypeError(f"{source_key} is not a torch.Tensor")
            lstm_state[target_key] = tensor.detach().cpu()

    fc_weight = state.get("fc_head.weight")
    fc_bias = state.get("fc_head.bias")
    if not torch.is_tensor(fc_weight):
        raise TypeError("fc_head.weight is missing or is not a torch.Tensor")
    if not torch.is_tensor(fc_bias):
        raise TypeError("fc_head.bias is missing or is not a torch.Tensor")

    lstm.load_state_dict(lstm_state, strict=True)
    fc.load_state_dict(
        {
            "weight": fc_weight.detach().cpu(),
            "bias": fc_bias.detach().cpu(),
        },
        strict=True,
    )

    lstm = lstm.to("cpu")
    fc = fc.to("cpu")
    lstm.eval()
    fc.eval()
    return lstm, fc


def _verify_forward(
    checkpoint: Mapping[str, object],
    exported: ExportedText,
    cfg: ExportConfig,
    experiment_root: Path,
) -> Tuple[float, float]:
    # experiment_root is retained in the public helper signature for backward
    # compatibility with main(), but architecture verification no longer depends
    # on 10_1_config.py / 10_4_models.py.
    del experiment_root

    p = exported.params
    lstm, fc = _build_torch_reference_from_checkpoint(
        checkpoint=checkpoint,
        params=p,
    )
    rng = np.random.default_rng(cfg.VERIFY_RANDOM_SEED)

    # Generate normalized-space values in a moderate range, then map them back
    # to physical space. This gives valid numerical scales without requiring H5.
    x_norm = rng.normal(
        loc=0.0,
        scale=0.5,
        size=(
            int(cfg.VERIFY_BATCH_SIZE),
            int(cfg.VERIFY_SEQUENCE_LENGTH),
            p.n_in,
        ),
    )
    x_physical = (
        x_norm * exported.norm.input_std[None, None, :]
        + exported.norm.input_mean[None, None, :]
    )

    with torch.no_grad():
        x_t = torch.as_tensor(
            x_norm,
            dtype=torch.float32,
        )
        lstm_out_t, _ = lstm(x_t)
        pred_norm_t = fc(lstm_out_t)
        pred_physical_t = (
            pred_norm_t
            * torch.as_tensor(
                exported.norm.output_std,
                dtype=torch.float32,
            )
            + torch.as_tensor(
                exported.norm.output_mean,
                dtype=torch.float32,
            )
        )
        pred_torch = (
            pred_physical_t
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    pred_fortran = _fortran_style_forward_sequence(
        x_physical,
        exported,
    )

    abs_diff = np.abs(pred_fortran - pred_torch)
    max_abs = float(np.max(abs_diff))
    denom = np.maximum(
        np.abs(pred_torch),
        1.0e-12,
    )
    max_rel = float(
        np.max(abs_diff / denom)
    )

    if not np.allclose(
        pred_fortran,
        pred_torch,
        atol=cfg.FORWARD_ATOL,
        rtol=cfg.FORWARD_RTOL,
    ):
        idx = np.unravel_index(
            int(np.argmax(abs_diff)),
            abs_diff.shape,
        )
        raise AssertionError(
            "Forward verification FAILED.\n"
            f"max_abs={max_abs:.6e}, max_rel={max_rel:.6e}\n"
            f"index={idx}, fortran_emulation={pred_fortran[idx]:.17e}, "
            f"pytorch={pred_torch[idx]:.17e}\n"
            "Likely causes: matrix ordering, checkpoint tensor mismatch, "
            "normalization mismatch, or unsupported architecture."
        )

    return max_abs, max_rel


# =============================================================================
# 4. Main
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export Step-10 best_checkpoint.pt to the text format read by "
            "Abaqus/Fortran mlmodule.f."
        )
    )
    parser.add_argument(
        "--experiment-root",
        type=str,
        default=None,
        help="Override CFG.EXPERIMENT_ROOT.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional explicit output txt path. "
            "Default: <experiment>/exports/BCC_best.txt or HCP_best.txt"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional explicit checkpoint path.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default=None,
        help="Optional explicit normalization npz path.",
    )
    parser.add_argument(
        "--skip-forward-check",
        action="store_true",
        help="Skip PyTorch-vs-Fortran forward verification.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    experiment_root = Path(
        args.experiment_root
        if args.experiment_root
        else CFG.EXPERIMENT_ROOT
    )

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else experiment_root / CFG.CHECKPOINT_RELATIVE_PATH
    )
    normalization_path = (
        Path(args.normalization)
        if args.normalization
        else experiment_root / CFG.NORMALIZATION_RELATIVE_PATH
    )

    print("=" * 100)
    print("Step-10 -> Fortran LSTM text exporter")
    print("=" * 100)
    print("Experiment root :", experiment_root)
    print("Checkpoint      :", checkpoint_path)
    print("Normalization   :", normalization_path)

    checkpoint = _load_checkpoint(checkpoint_path)
    raw_state = checkpoint["model_state_dict"]
    if not isinstance(raw_state, Mapping):
        raise TypeError("model_state_dict is not a mapping")

    # Narrow type for internal helpers.
    state: Dict[str, torch.Tensor] = dict(raw_state)

    norm = _load_normalization(normalization_path)
    params = _infer_parameters(state)

    phase = _validate_fortran_compatibility(
        params=params,
        norm=norm,
        checkpoint=checkpoint,
        cfg=CFG,
    )

    if args.output:
        output_path = Path(args.output)
    elif CFG.OUTPUT_NAME.lower().strip() != "auto":
        output_path = (
            experiment_root
            / CFG.EXPORT_SUBDIR
            / CFG.OUTPUT_NAME
        )
    else:
        output_path = (
            experiment_root
            / CFG.EXPORT_SUBDIR
            / f"{phase}_best.txt"
        )

    print("Phase           :", phase)
    print("Input dim       :", params.n_in)
    print("Hidden size     :", params.n_hidden)
    print("Layers          :", params.n_layers)
    print("Output dim      :", params.n_out)
    if checkpoint.get("multiplier") is not None:
        print("Multiplier      :", checkpoint.get("multiplier"))
    if norm.sequence_length is not None:
        print("Sequence length :", norm.sequence_length)

    _export_txt(
        output_path=output_path,
        params=params,
        norm=norm,
        cfg=CFG,
    )

    print("Exported txt    :", output_path)
    print(
        "Size            :",
        f"{output_path.stat().st_size / (1024*1024):.3f} MiB",
    )

    exported = _read_exported_txt(output_path)

    if CFG.VERIFY_READBACK:
        _verify_readback(
            exported=exported,
            params=params,
            norm=norm,
            cfg=CFG,
        )
        print("[PASS] txt write/read-back tensor check")

    if CFG.VERIFY_FORWARD and not args.skip_forward_check:
        max_abs, max_rel = _verify_forward(
            checkpoint=checkpoint,
            exported=exported,
            cfg=CFG,
            experiment_root=experiment_root,
        )
        print(
            "[PASS] PyTorch vs Fortran-emulation forward check"
        )
        print("       max_abs_error =", f"{max_abs:.6e}")
        print("       max_rel_error =", f"{max_rel:.6e}")
    else:
        print("[SKIP] forward verification")

    print("=" * 100)
    print("READY FOR FORTRAN:")
    print(output_path)
    print("=" * 100)


if __name__ == "__main__":
    main()
