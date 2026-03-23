#!/usr/bin/env python3
"""Rebuild patched recurrent state-update stage variants for A3 runtime bisection."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from ptodsl.compiler.jit import _discover_include_dirs


PATCH_NO_EXP = (
    "    TEXP(v112, v112);\n",
    "    /* bisect: skip TEXP on scalar-shaped tile */\n",
)

PATCH_NO_EXTRACT_TERM = (
    "    wait_flag(PIPE_V, PIPE_MTE1, EVENT_ID0);\n"
    "    TEXTRACT(v116, v76, v16, v16);\n"
    "    set_flag(PIPE_MTE1, PIPE_V, EVENT_ID0);\n"
    "    wait_flag(PIPE_MTE1, PIPE_V, EVENT_ID0);\n"
    "    TROWEXPAND(v100, v116);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    TMUL(v88, v100, v64);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    TMUL(v88, v88, v96);\n"
    "    TMUL(v84, v80, v92);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    TADD(v84, v84, v88);\n"
    "    pipe_barrier(PIPE_V);\n",
    "    TMUL(v84, v80, v92);\n"
    "    pipe_barrier(PIPE_V);\n",
)

PATCH_COPY_STATE = (
    "    wait_flag(PIPE_V, PIPE_MTE1, EVENT_ID0);\n"
    "    TEXTRACT(v116, v76, v16, v16);\n"
    "    set_flag(PIPE_MTE1, PIPE_V, EVENT_ID0);\n"
    "    wait_flag(PIPE_MTE1, PIPE_V, EVENT_ID0);\n"
    "    TROWEXPAND(v100, v116);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    TMUL(v88, v100, v64);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    TMUL(v88, v88, v96);\n"
    "    TMUL(v84, v80, v92);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    TADD(v84, v84, v88);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);\n"
    "    TCVT(v56, v84, v8);\n"
    "    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);\n"
    "    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);\n"
    "    pipe_barrier(PIPE_MTE3);\n"
    "    TSTORE(v141, v56);\n"
    "    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);\n",
    "    wait_flag(PIPE_V, PIPE_MTE1, EVENT_ID0);\n"
    "    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);\n"
    "    pipe_barrier(PIPE_MTE3);\n"
    "    TSTORE(v141, v52);\n"
    "    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);\n",
)

PATCH_COPY_STATE_ROUNDTRIP = (
    "    wait_flag(PIPE_V, PIPE_MTE1, EVENT_ID0);\n"
    "    TEXTRACT(v116, v76, v16, v16);\n"
    "    set_flag(PIPE_MTE1, PIPE_V, EVENT_ID0);\n"
    "    wait_flag(PIPE_MTE1, PIPE_V, EVENT_ID0);\n"
    "    TROWEXPAND(v100, v116);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    TMUL(v88, v100, v64);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    TMUL(v88, v88, v96);\n"
    "    TMUL(v84, v80, v92);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    TADD(v84, v84, v88);\n"
    "    pipe_barrier(PIPE_V);\n",
    "    TCVT(v84, v52, v8);\n"
    "    pipe_barrier(PIPE_V);\n",
)

ORIGINAL_MAIN_BLOCK = (
    "    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);\n"
    "    TCVT(v108, v104, v8);\n"
    "    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);\n"
    "    TEXP(v112, v112);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    TROWEXPAND(v92, v112);\n"
    "    TROWEXPAND(v96, v108);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    TMUL(v72, v68, v92);\n"
    "    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    TSUB(v76, v60, v72);\n"
    "    set_flag(PIPE_V, PIPE_MTE1, EVENT_ID0);\n"
    "    int32_t v135 = (int32_t) ((uint32_t) v119 * (uint32_t) v13);\n"
    "    pto::Shape<1, 1, 1, 1, 16> v136 = pto::Shape<1, 1, 1, 1, 16>();\n"
    "    pto::Stride<16, 16, 16, 16, 1> v137 = pto::Stride<16, 16, 16, 16, 1>();\n"
    "    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v138 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v135 * (unsigned) v13 + v11 * (unsigned) v15), v136, v137);\n"
    "    pto::Shape<1, 1, 1, 1, 16> v139 = pto::Shape<1, 1, 1, 1, 16>();\n"
    "    pto::Stride<16, 16, 16, 16, 1> v140 = pto::Stride<16, 16, 16, 16, 1>();\n"
    "    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v141 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v135 * (unsigned) v13 + v11 * (unsigned) v15), v139, v140);\n"
    "    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);\n"
    "    TLOAD(v52, v138);\n"
    "    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);\n"
    "    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);\n"
    "    TCVT(v80, v52, v8);\n"
    "    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);\n"
    "    wait_flag(PIPE_V, PIPE_MTE1, EVENT_ID0);\n"
    "    TEXTRACT(v116, v76, v16, v16);\n"
    "    set_flag(PIPE_MTE1, PIPE_V, EVENT_ID0);\n"
    "    wait_flag(PIPE_MTE1, PIPE_V, EVENT_ID0);\n"
    "    TROWEXPAND(v100, v116);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    TMUL(v88, v100, v64);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    TMUL(v88, v88, v96);\n"
    "    TMUL(v84, v80, v92);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    TADD(v84, v84, v88);\n"
    "    pipe_barrier(PIPE_V);\n"
    "    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);\n"
    "    TCVT(v56, v84, v8);\n"
    "    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);\n"
    "    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);\n"
    "    pipe_barrier(PIPE_MTE3);\n"
    "    TSTORE(v141, v56);\n"
    "    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);\n"
)

PATCH_MINIMAL_STATE_STORE = (
    ORIGINAL_MAIN_BLOCK,
    "    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);\n"
    "    int32_t v135 = (int32_t) ((uint32_t) v119 * (uint32_t) v13);\n"
    "    pto::Shape<1, 1, 1, 1, 16> v136 = pto::Shape<1, 1, 1, 1, 16>();\n"
    "    pto::Stride<16, 16, 16, 16, 1> v137 = pto::Stride<16, 16, 16, 16, 1>();\n"
    "    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v138 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v135 * (unsigned) v13 + v11 * (unsigned) v15), v136, v137);\n"
    "    pto::Shape<1, 1, 1, 1, 16> v139 = pto::Shape<1, 1, 1, 1, 16>();\n"
    "    pto::Stride<16, 16, 16, 16, 1> v140 = pto::Stride<16, 16, 16, 16, 1>();\n"
    "    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v141 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v135 * (unsigned) v13 + v11 * (unsigned) v15), v139, v140);\n"
    "    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);\n"
    "    TLOAD(v52, v138);\n"
    "    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);\n"
    "    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);\n"
    "    pipe_barrier(PIPE_MTE3);\n"
    "    TSTORE(v141, v52);\n"
    "    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);\n",
)

PATCH_MINIMAL_STATE_ROUNDTRIP = (
    ORIGINAL_MAIN_BLOCK,
    "    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);\n"
    "    int32_t v135 = (int32_t) ((uint32_t) v119 * (uint32_t) v13);\n"
    "    pto::Shape<1, 1, 1, 1, 16> v136 = pto::Shape<1, 1, 1, 1, 16>();\n"
    "    pto::Stride<16, 16, 16, 16, 1> v137 = pto::Stride<16, 16, 16, 16, 1>();\n"
    "    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v138 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v135 * (unsigned) v13 + v11 * (unsigned) v15), v136, v137);\n"
    "    pto::Shape<1, 1, 1, 1, 16> v139 = pto::Shape<1, 1, 1, 1, 16>();\n"
    "    pto::Stride<16, 16, 16, 16, 1> v140 = pto::Stride<16, 16, 16, 16, 1>();\n"
    "    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v141 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v135 * (unsigned) v13 + v11 * (unsigned) v15), v139, v140);\n"
    "    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);\n"
    "    TLOAD(v52, v138);\n"
    "    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);\n"
    "    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);\n"
    "    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);\n"
    "    TCVT(v80, v52, v8);\n"
    "    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);\n"
    "    TCVT(v56, v80, v8);\n"
    "    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);\n"
    "    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);\n"
    "    pipe_barrier(PIPE_MTE3);\n"
    "    TSTORE(v141, v56);\n"
    "    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);\n",
)


def _compile_shared_library(*, stage_dir: Path, caller_cpp_path: Path, lib_path: Path, npu_arch: str) -> None:
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
    for include_dir in include_dirs:
        cmd.insert(1, f"-I{include_dir}")
    (stage_dir / "compile_cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    subprocess.run(cmd, check=True, cwd=str(stage_dir))


def _patch_text(text: str, mode: str) -> str:
    patched = text
    replacements = []
    if mode in {"no_exp", "no_exp_no_extract"}:
        replacements.append(PATCH_NO_EXP)
    if mode in {"no_extract_term", "no_exp_no_extract"}:
        replacements.append(PATCH_NO_EXTRACT_TERM)
    if mode == "copy_state":
        replacements.append(PATCH_COPY_STATE)
    if mode == "copy_state_roundtrip":
        replacements.append(PATCH_COPY_STATE_ROUNDTRIP)
    if mode == "minimal_state_store":
        replacements.append(PATCH_MINIMAL_STATE_STORE)
    if mode == "minimal_state_roundtrip":
        replacements.append(PATCH_MINIMAL_STATE_ROUNDTRIP)
    for old, new in replacements:
        if old not in patched:
            raise RuntimeError(f"Failed to find expected snippet for mode {mode!r}.")
        patched = patched.replace(old, new, 1)
    return patched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("/tmp/recurrent_row_specialized"),
        help="Directory containing the original staged recurrent artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to place the patched stage artifacts.",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "no_exp",
            "no_extract_term",
            "no_exp_no_extract",
            "copy_state",
            "copy_state_roundtrip",
            "minimal_state_store",
            "minimal_state_roundtrip",
        ),
        required=True,
        help="Patch mode to apply to stage_state_update_row_000/kernel.cpp.",
    )
    parser.add_argument("--npu-arch", default="dav-2201")
    args = parser.parse_args()

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    shutil.copytree(args.artifacts_dir, args.output_dir)

    stage_dir = args.output_dir / "stage_state_update_row_000"
    kernel_cpp_path = stage_dir / "kernel.cpp"
    patched = _patch_text(kernel_cpp_path.read_text(encoding="utf-8"), args.mode)
    kernel_cpp_path.write_text(patched, encoding="utf-8")
    (stage_dir / "patch_mode.txt").write_text(args.mode + "\n", encoding="utf-8")
    _compile_shared_library(
        stage_dir=stage_dir,
        caller_cpp_path=stage_dir / "caller.cpp",
        lib_path=stage_dir / "kernel.so",
        npu_arch=args.npu_arch,
    )
    print(f"rebuilt {stage_dir / 'kernel.so'} with mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
