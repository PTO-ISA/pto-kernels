#!/usr/bin/env python3
"""Direct runtime repro for recurrent_gated_delta_rule staged A3 kernels.

This bypasses PTODSL rebuild and the multi-stage Python wrapper. It loads a
prebuilt stage `kernel.so` directly and calls the generated `call_kernel(...)`
entrypoint with real NPU tensors for the smoke variant.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from pto_kernels.ops.attention.recurrent_gated_delta_rule.runtime import (
    RecurrentGatedDeltaRuleVariant,
    make_recurrent_gated_delta_rule_inputs,
)


SMOKE_VARIANT = RecurrentGatedDeltaRuleVariant(seq_len=2, dim=16, seed=0)


def _tensor_ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _stream_ptr():
    return torch.npu.current_stream()._as_parameter_


def _load_lib(lib_path: Path, argc: int):
    lib = ctypes.CDLL(str(lib_path))
    lib.call_kernel.restype = None
    lib.call_kernel.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * argc
    return lib


def _call_proj(artifacts_dir: Path, block_dim: int, inputs: dict[str, object]) -> dict[str, object]:
    lib = _load_lib(artifacts_dir / "stage_proj" / "kernel.so", 3)
    rows = SMOKE_VARIANT.total_tokens * SMOKE_VARIANT.num_value_heads
    proj = torch.empty((rows, SMOKE_VARIANT.dim), dtype=torch.float32, device=inputs["query"].device)
    start = time.perf_counter()
    lib.call_kernel(
        ctypes.c_uint32(block_dim),
        _stream_ptr(),
        _tensor_ptr(proj),
        _tensor_ptr(inputs["state"]),
        _tensor_ptr(inputs["key"].reshape(rows, SMOKE_VARIANT.dim)),
    )
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {"proj": proj, "elapsed_ms": elapsed_ms}


def _call_state_update(
    artifacts_dir: Path,
    block_dim: int,
    inputs: dict[str, object],
    proj: torch.Tensor,
) -> dict[str, object]:
    lib = _load_lib(artifacts_dir / "stage_state_update_row_000" / "kernel.so", 7)
    rows = SMOKE_VARIANT.total_tokens * SMOKE_VARIANT.num_value_heads
    state_out = torch.empty_like(inputs["state"])
    start = time.perf_counter()
    lib.call_kernel(
        ctypes.c_uint32(block_dim),
        _stream_ptr(),
        _tensor_ptr(state_out.reshape(rows * SMOKE_VARIANT.dim, SMOKE_VARIANT.dim)),
        _tensor_ptr(inputs["state"].reshape(rows * SMOKE_VARIANT.dim, SMOKE_VARIANT.dim)),
        _tensor_ptr(proj),
        _tensor_ptr(inputs["value"].reshape(rows, SMOKE_VARIANT.dim)),
        _tensor_ptr(inputs["key"].reshape(rows, SMOKE_VARIANT.dim)),
        _tensor_ptr(inputs["beta"].reshape(rows, 1)),
        _tensor_ptr(inputs["g"].reshape(rows, 1)),
    )
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {"state_out": state_out, "elapsed_ms": elapsed_ms}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("/tmp/recurrent_row_specialized"),
        help="Directory containing prebuilt staged kernel artifacts.",
    )
    parser.add_argument(
        "--stage",
        choices=("proj", "state_update_row_000"),
        required=True,
        help="Which prebuilt stage to run directly.",
    )
    parser.add_argument("--block-dim", type=int, default=1)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    torch.npu.set_device("npu:0")
    inputs = make_recurrent_gated_delta_rule_inputs(SMOKE_VARIANT, device_index=0)

    result: dict[str, object] = {
        "artifacts_dir": str(args.artifacts_dir),
        "stage": args.stage,
        "block_dim": args.block_dim,
        "variant": SMOKE_VARIANT.as_dict(),
        "shape_summary": SMOKE_VARIANT.shape_summary,
    }

    proj_result = _call_proj(args.artifacts_dir, args.block_dim, inputs)
    result["stage_proj_elapsed_ms"] = proj_result["elapsed_ms"]

    if args.stage == "proj":
        result["status"] = "ok"
    else:
        state_result = _call_state_update(args.artifacts_dir, args.block_dim, inputs, proj_result["proj"])
        result["stage_state_update_row_000_elapsed_ms"] = state_result["elapsed_ms"]
        result["status"] = "ok"

    report_path = args.report or (args.artifacts_dir / f"repro_{args.stage}.json")
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
