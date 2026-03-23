#!/usr/bin/env python3
"""Hand-written A3 vector state row load/store repro for recurrent debugging."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from ptodsl.compiler.jit import _discover_include_dirs

from pto_kernels.ops.attention.recurrent_gated_delta_rule.runtime import (
    RecurrentGatedDeltaRuleVariant,
    make_recurrent_gated_delta_rule_inputs,
)


SMOKE_VARIANT = RecurrentGatedDeltaRuleVariant(seq_len=2, dim=16, seed=0)

DTYPE_CPP = {
    "bf16": "bfloat16_t",
    "fp16": "half",
}

DTYPE_TORCH = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

KERNEL_CPP_LOOP = r'''#include "pto/pto-inst.hpp"
using namespace pto;

__global__ AICORE void recurrent_manual_state_store(__gm__ {DTYPE}* out_ptr, __gm__ {DTYPE}* in_ptr) {
  int32_t cRows = 32;
  int32_t cD = 16;
  int32_t c1 = 1;
  int32_t c0 = 0;

#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  int64_t bid64 = get_block_idx();
  int64_t blocks64 = get_block_num();
  int32_t bid = (int32_t)bid64;
  int32_t num_blocks = (int32_t)blocks64;
  int32_t rows_per_core = cRows / num_blocks;
  if ((cRows % num_blocks) != c0 && (cRows < c0) == (num_blocks < c0)) {
    rows_per_core = rows_per_core + c1;
  }
  int32_t row_start = (int32_t)((uint32_t)bid * (uint32_t)rows_per_core);
  int32_t row_end = (int32_t)((uint32_t)row_start + (uint32_t)rows_per_core);
  if ((uint32_t)row_end > (uint32_t)cRows) {
    row_end = cRows;
  }

  Tile<TileType::Vec, {DTYPE}, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> ub;
  TASSIGN(ub, (int64_t)0);
  Tile<TileType::Vec, {DTYPE}, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> row_tile;
  __ubuf__ {DTYPE}* row_ptr = ub.data();
  uint64_t row_addr = reinterpret_cast<uint64_t>(row_ptr);
  TASSIGN(row_tile, row_addr);

  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

  for (int32_t row = row_start; row < row_end; row += c1) {
    int32_t state_base = (int32_t)((uint32_t)row * (uint32_t)cD);
    pto::Shape<1, 1, 1, 1, 16> shape = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> stride = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<{DTYPE}, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>
      src(in_ptr + (c0 + (unsigned)state_base * (unsigned)cD + c0 * (unsigned)c1), shape, stride);
    GlobalTensor<{DTYPE}, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>
      dst(out_ptr + (c0 + (unsigned)state_base * (unsigned)cD + c0 * (unsigned)c1), shape, stride);

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(row_tile, src);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(dst, row_tile);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }

  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
#endif
}
'''

KERNEL_CPP_SINGLE_ROW = r'''#include "pto/pto-inst.hpp"
using namespace pto;

__global__ AICORE void recurrent_manual_state_store(__gm__ {DTYPE}* out_ptr, __gm__ {DTYPE}* in_ptr) {
  int32_t c1 = 1;
  int32_t c0 = 0;

#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  Tile<TileType::Vec, {DTYPE}, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> ub;
  TASSIGN(ub, (int64_t)0);
  Tile<TileType::Vec, {DTYPE}, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> row_tile;
  __ubuf__ {DTYPE}* row_ptr = ub.data();
  uint64_t row_addr = reinterpret_cast<uint64_t>(row_ptr);
  TASSIGN(row_tile, row_addr);

  pto::Shape<1, 1, 1, 1, 16> shape = pto::Shape<1, 1, 1, 1, 16>();
  pto::Stride<16, 16, 16, 16, 1> stride = pto::Stride<16, 16, 16, 16, 1>();
  GlobalTensor<{DTYPE}, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>
    src(in_ptr + (c0 + (unsigned)c0 * (unsigned)16 + c0 * (unsigned)c1), shape, stride);
  GlobalTensor<{DTYPE}, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>
    dst(out_ptr + (c0 + (unsigned)c0 * (unsigned)16 + c0 * (unsigned)c1), shape, stride);

  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  TLOAD(row_tile, src);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  pipe_barrier(PIPE_MTE3);
  TSTORE(dst, row_tile);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
#endif
}
'''

KERNEL_CPP_SINGLE_ROW_LOAD_ONLY = r'''#include "pto/pto-inst.hpp"
using namespace pto;

__global__ AICORE void recurrent_manual_state_store(__gm__ {DTYPE}* out_ptr, __gm__ {DTYPE}* in_ptr) {
  int32_t c1 = 1;
  int32_t c0 = 0;

#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  Tile<TileType::Vec, {DTYPE}, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> ub;
  TASSIGN(ub, (int64_t)0);
  Tile<TileType::Vec, {DTYPE}, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> row_tile;
  __ubuf__ {DTYPE}* row_ptr = ub.data();
  uint64_t row_addr = reinterpret_cast<uint64_t>(row_ptr);
  TASSIGN(row_tile, row_addr);

  pto::Shape<1, 1, 1, 1, 16> shape = pto::Shape<1, 1, 1, 1, 16>();
  pto::Stride<16, 16, 16, 16, 1> stride = pto::Stride<16, 16, 16, 16, 1>();
  GlobalTensor<{DTYPE}, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>
    src(in_ptr + (c0 + (unsigned)c0 * (unsigned)16 + c0 * (unsigned)c1), shape, stride);

  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  TLOAD(row_tile, src);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
#endif
}
'''

KERNEL_CPP_SINGLE_ROW_STORE_ONLY = r'''#include "pto/pto-inst.hpp"
using namespace pto;

__global__ AICORE void recurrent_manual_state_store(__gm__ {DTYPE}* out_ptr, __gm__ {DTYPE}* in_ptr) {
  int32_t c1 = 1;
  int32_t c0 = 0;

#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  Tile<TileType::Vec, {DTYPE}, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> ub;
  TASSIGN(ub, (int64_t)0);
  Tile<TileType::Vec, {DTYPE}, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> row_tile;
  __ubuf__ {DTYPE}* row_ptr = ub.data();
  uint64_t row_addr = reinterpret_cast<uint64_t>(row_ptr);
  TASSIGN(row_tile, row_addr);

  pto::Shape<1, 1, 1, 1, 16> shape = pto::Shape<1, 1, 1, 1, 16>();
  pto::Stride<16, 16, 16, 16, 1> stride = pto::Stride<16, 16, 16, 16, 1>();
  GlobalTensor<{DTYPE}, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>
    dst(out_ptr + (c0 + (unsigned)c0 * (unsigned)16 + c0 * (unsigned)c1), shape, stride);

  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  pipe_barrier(PIPE_MTE3);
  TSTORE(dst, row_tile);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
#endif
}
'''

KERNEL_CPP_SINGLE_ROW_ROUNDTRIP = r'''#include "pto/pto-inst.hpp"
using namespace pto;

__global__ AICORE void recurrent_manual_state_store(__gm__ {DTYPE}* out_ptr, __gm__ {DTYPE}* in_ptr) {
  int32_t c1 = 1;
  int32_t c0 = 0;
  RoundMode round_mode = RoundMode::CAST_ROUND;

#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  Tile<TileType::Vec, {DTYPE}, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> ub_in;
  TASSIGN(ub_in, (int64_t)0);
  Tile<TileType::Vec, {DTYPE}, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> row_in;
  __ubuf__ {DTYPE}* in_ptr_ub = ub_in.data();
  uint64_t in_addr = reinterpret_cast<uint64_t>(in_ptr_ub);
  TASSIGN(row_in, in_addr);

  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> ub_fp32;
  TASSIGN(ub_fp32, (int64_t)288);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> row_fp32;
  __ubuf__ float* fp32_ptr = ub_fp32.data();
  uint64_t fp32_addr = reinterpret_cast<uint64_t>(fp32_ptr);
  TASSIGN(row_fp32, fp32_addr);

  Tile<TileType::Vec, {DTYPE}, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> ub_out;
  TASSIGN(ub_out, (int64_t)608);
  Tile<TileType::Vec, {DTYPE}, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> row_out;
  __ubuf__ {DTYPE}* out_ptr_ub = ub_out.data();
  uint64_t out_addr = reinterpret_cast<uint64_t>(out_ptr_ub);
  TASSIGN(row_out, out_addr);

  pto::Shape<1, 1, 1, 1, 16> shape = pto::Shape<1, 1, 1, 1, 16>();
  pto::Stride<16, 16, 16, 16, 1> stride = pto::Stride<16, 16, 16, 16, 1>();
  GlobalTensor<{DTYPE}, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>
    src(in_ptr + (c0 + (unsigned)c0 * (unsigned)16 + c0 * (unsigned)c1), shape, stride);
  GlobalTensor<{DTYPE}, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>
    dst(out_ptr + (c0 + (unsigned)c0 * (unsigned)16 + c0 * (unsigned)c1), shape, stride);

  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  TLOAD(row_in, src);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(row_fp32, row_in, round_mode);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  TCVT(row_out, row_fp32, round_mode);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  pipe_barrier(PIPE_MTE3);
  TSTORE(dst, row_out);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
#endif
}
'''

CALLER_CPP = r'''#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *in_ptr)
{
    recurrent_manual_state_store<<<blockDim, nullptr, stream>>>(({DTYPE} *)out_ptr, ({DTYPE} *)in_ptr);
}
'''


def _compile_shared_library(stage_dir: Path, caller_cpp_path: Path, lib_path: Path, npu_arch: str) -> None:
    toolkit_home = os.environ.get("ASCEND_TOOLKIT_HOME")
    if not toolkit_home:
        raise RuntimeError("ASCEND_TOOLKIT_HOME is required.")
    include_dirs = _discover_include_dirs(toolkit_home)
    cmd = [
        "bisheng",
        "-fPIC",
        "-shared",
        "-D_FORTIFY_SOURCE=2",
        "-O2",
        "-std=c++17",
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        "-fstack-protector-strong",
        "-xcce",
        "-Xhost-start",
        "-Xhost-end",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=true",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        f"--npu-arch={npu_arch}",
        "-DMEMORY_BASE",
        "-std=gnu++17",
        str(caller_cpp_path),
        f"-L{toolkit_home}/lib64",
        f"-Wl,-rpath,{toolkit_home}/lib64",
        "-lruntime",
        "-o",
        str(lib_path),
    ]
    for include_dir in _discover_include_dirs(toolkit_home):
        cmd.insert(1, f"-I{include_dir}")
    (stage_dir / "compile_cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    subprocess.run(cmd, check=True, cwd=str(stage_dir))


def _tensor_ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _stream_ptr():
    return torch.npu.current_stream()._as_parameter_


def _render_kernel(source: str, dtype_name: str) -> str:
    return source.replace("{DTYPE}", DTYPE_CPP[dtype_name])


def _build_kernel(output_dir: Path, npu_arch: str, mode: str, dtype_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_cpp = output_dir / "kernel.cpp"
    caller_cpp = output_dir / "caller.cpp"
    kernel_so = output_dir / "kernel.so"
    if mode == "single_row":
        kernel_src = KERNEL_CPP_SINGLE_ROW
    elif mode == "single_row_load_only":
        kernel_src = KERNEL_CPP_SINGLE_ROW_LOAD_ONLY
    elif mode == "single_row_store_only":
        kernel_src = KERNEL_CPP_SINGLE_ROW_STORE_ONLY
    elif mode == "single_row_roundtrip":
        kernel_src = KERNEL_CPP_SINGLE_ROW_ROUNDTRIP
    else:
        kernel_src = KERNEL_CPP_LOOP
    kernel_cpp.write_text(_render_kernel(kernel_src, dtype_name), encoding="utf-8")
    caller_cpp.write_text(_render_kernel(CALLER_CPP, dtype_name), encoding="utf-8")
    _compile_shared_library(output_dir, caller_cpp, kernel_so, npu_arch)
    return kernel_so


def _load_lib(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    lib.call_kernel.restype = None
    lib.call_kernel.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    return lib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/recurrent_manual_state_store"))
    parser.add_argument("--block-dim", type=int, default=1)
    parser.add_argument("--npu-arch", default="dav-2201")
    parser.add_argument(
        "--mode",
        choices=("loop", "single_row", "single_row_load_only", "single_row_store_only", "single_row_roundtrip"),
        default="loop",
    )
    parser.add_argument("--dtype", choices=tuple(sorted(DTYPE_CPP)), default="bf16")
    args = parser.parse_args()

    torch.npu.set_device("npu:0")
    inputs = make_recurrent_gated_delta_rule_inputs(SMOKE_VARIANT, device_index=0)
    torch_dtype = DTYPE_TORCH[args.dtype]
    state_in = inputs["state"].to(torch_dtype).reshape(
        SMOKE_VARIANT.total_tokens * SMOKE_VARIANT.num_value_heads * SMOKE_VARIANT.dim,
        SMOKE_VARIANT.dim,
    )
    state_out = torch.empty_like(state_in)

    lib_path = _build_kernel(args.output_dir, args.npu_arch, args.mode, args.dtype)
    lib = _load_lib(lib_path)

    start = time.perf_counter()
    lib.call_kernel(
        ctypes.c_uint32(args.block_dim),
        _stream_ptr(),
        _tensor_ptr(state_out),
        _tensor_ptr(state_in),
    )
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    report = {
        "output_dir": str(args.output_dir),
        "kernel_so": str(lib_path),
        "block_dim": args.block_dim,
        "mode": args.mode,
        "dtype": args.dtype,
        "elapsed_ms": elapsed_ms,
        "max_abs_diff": float((state_out - state_in).abs().max().item()),
        "status": "ok",
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
