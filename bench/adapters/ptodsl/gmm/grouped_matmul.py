from __future__ import annotations

import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

from pto_kernels.bench.adapter_utils import compile_pto_kernel, describe_pto
from pto_kernels.ops.gmm.grouped_matmul.runtime import (
    VARIANT,
    make_dense_single_weight_inputs,
    run_pto_dense_variant,
)


KERNEL = "python/pto_kernels/ops/gmm/grouped_matmul/kernel.py"
META = "python/pto_kernels/ops/gmm/grouped_matmul/meta.py"


def describe(repo_root, spec):
    return describe_pto(repo_root, KERNEL, META)


def compile_kernel(repo_root, spec, artifacts_dir):
    return compile_pto_kernel(repo_root, KERNEL, artifacts_dir)


def benchmark(repo_root, spec, artifacts_dir):
    kernel_file = repo_root / KERNEL
    spec_obj = importlib.util.spec_from_file_location("pto_grouped_matmul_kernel", kernel_file)
    if spec_obj is None or spec_obj.loader is None:
        return {"status": "blocked", "reason": f"Unable to import {kernel_file}"}

    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    wrapper = module.build_jit_wrapper(output_dir=artifacts_dir)
    build = getattr(wrapper, "_build", None)
    try:
        if callable(build):
            build()
    except Exception as exc:  # pragma: no cover - exercised on NPU bring-up hosts
        report = {
            "status": "blocked",
            "variant": VARIANT.as_dict(),
            "reason": f"PTO compile failed: {exc}",
        }
        report_path = Path(artifacts_dir) / "ptodsl_grouped_matmul_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    inputs = make_dense_single_weight_inputs(device_index=int(spec.device.get("id", 0)))
    reference = inputs["baseline_reference"]

    try:
        for _ in range(spec.bench.warmup):
            run_pto_dense_variant(wrapper, inputs)
        torch.npu.synchronize()

        timings_ms = []
        output = None
        for _ in range(spec.bench.repeat):
            torch.npu.synchronize()
            start = time.perf_counter()
            output = run_pto_dense_variant(wrapper, inputs)
            torch.npu.synchronize()
            timings_ms.append((time.perf_counter() - start) * 1000.0)
    except Exception as exc:  # pragma: no cover - exercised on NPU bring-up hosts
        report = {
            "status": "blocked",
            "variant": VARIANT.as_dict(),
            "reason": f"PTO execution failed: {exc}",
            "artifact_paths": [str(path) for path in getattr(wrapper, "_artifact_paths", lambda: ())()],
        }
        report_path = Path(artifacts_dir) / "ptodsl_grouped_matmul_benchmark.json"
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
    report_path = Path(artifacts_dir) / "ptodsl_grouped_matmul_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
