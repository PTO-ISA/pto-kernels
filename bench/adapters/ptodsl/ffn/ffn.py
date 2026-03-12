from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
from pto_kernels.ops.ffn.ffn.runtime import (
    VARIANT,
    make_dense_relu_inputs,
    run_pto_ffn_variant,
)

from pto_kernels.bench.adapter_utils import compile_pto_kernel, describe_pto, load_module


KERNEL = "python/pto_kernels/ops/ffn/ffn/kernel.py"
META = "python/pto_kernels/ops/ffn/ffn/meta.py"


def describe(repo_root, spec):
    return describe_pto(repo_root, KERNEL, META)


def compile_kernel(repo_root, spec, artifacts_dir):
    return compile_pto_kernel(repo_root, KERNEL, artifacts_dir)


def benchmark(repo_root, spec, artifacts_dir):
    kernel_file = repo_root / KERNEL
    module = load_module(Path(kernel_file))
    builder = getattr(module, "build_jit_wrapper", None)
    if not callable(builder):
        return {"status": "blocked", "reason": "kernel module does not expose build_jit_wrapper(output_dir)"}

    wrapper = builder(output_dir=artifacts_dir)
    build = getattr(wrapper, "_build", None)
    try:
        if callable(build):
            build()
    except Exception as exc:  # pragma: no cover - exercised on NPU bring-up hosts
        report = {"status": "blocked", "reason": f"PTO compile failed: {exc}"}
        report_path = Path(artifacts_dir) / "ptodsl_ffn_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    inputs = make_dense_relu_inputs(device_index=int(spec.device.get("id", 0)))
    reference = inputs["reference"]

    try:
        for _ in range(spec.bench.warmup):
            run_pto_ffn_variant(wrapper, inputs)
        torch.npu.synchronize()

        timings_ms = []
        output = None
        for _ in range(spec.bench.repeat):
            torch.npu.synchronize()
            start = time.perf_counter()
            output = run_pto_ffn_variant(wrapper, inputs)
            torch.npu.synchronize()
            timings_ms.append((time.perf_counter() - start) * 1000.0)
    except Exception as exc:  # pragma: no cover - exercised on NPU bring-up hosts
        report = {
            "status": "blocked",
            "variant": VARIANT.as_dict(),
            "reason": f"PTO execution failed: {exc}",
            "artifact_paths": [str(path) for path in getattr(wrapper, "_artifact_paths", lambda: ())()],
        }
        report_path = Path(artifacts_dir) / "ptodsl_ffn_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    if output is None:
        return {"status": "blocked", "reason": "PTO benchmark did not produce an output tensor."}

    max_abs_diff = (output.float().cpu() - reference).abs().max().item()
    report = {
        "status": "ok",
        "variant": VARIANT.as_dict(),
        "timings_ms": {
            "median": statistics.median(timings_ms),
            "min": min(timings_ms),
            "max": max(timings_ms),
        },
        "correctness": {
            "max_abs_diff": max_abs_diff,
            "atol": spec.correctness.atol,
            "rtol": spec.correctness.rtol,
            "passes": bool(max_abs_diff <= spec.correctness.atol),
        },
        "artifact_paths": [str(path) for path in getattr(wrapper, "_artifact_paths", lambda: ())()],
    }
    report_path = Path(artifacts_dir) / "ptodsl_ffn_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
