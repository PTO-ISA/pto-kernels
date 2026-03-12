from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.gmm.grouped_matmul.runtime import (
    VARIANT,
    make_dense_single_weight_inputs,
    run_torch_npu_grouped_matmul,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "gmm", "grouped_matmul", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_grouped_matmul"
    summary["seed_variant"] = VARIANT.as_dict()
    summary["upstream_build_status"] = (
        "fast_kernel_launch_example is currently blocked on this host because bisheng "
        "fails while compiling torch header dependencies in the example extension."
    )
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    if not hasattr(torch_npu, "npu_grouped_matmul"):
        return {
            "status": "blocked",
            "reason": "torch_npu does not expose npu_grouped_matmul on this environment.",
        }
    return {
        "status": "runtime_builtin",
        "entrypoint": "torch_npu.npu_grouped_matmul",
        "note": (
            "Baseline execution currently relies on the runtime-installed op. "
            "Building the standalone ops-transformer example is tracked as a separate blocker."
        ),
    }


def benchmark(repo_root, spec, artifacts_dir):
    inputs = make_dense_single_weight_inputs(device_index=int(spec.device.get("id", 0)))
    reference = inputs["baseline_reference"]
    try:
        for _ in range(spec.bench.warmup):
            run_torch_npu_grouped_matmul(inputs)
        torch.npu.synchronize()

        timings_ms = []
        output = None
        for _ in range(spec.bench.repeat):
            torch.npu.synchronize()
            start = time.perf_counter()
            output = run_torch_npu_grouped_matmul(inputs)
            torch.npu.synchronize()
            timings_ms.append((time.perf_counter() - start) * 1000.0)
    except Exception as exc:  # pragma: no cover - exercised on NPU bring-up hosts
        report = {
            "status": "blocked",
            "variant": VARIANT.as_dict(),
            "entrypoint": "torch_npu.npu_grouped_matmul",
            "reason": str(exc),
            "blocking_gap": "ops-transformer-grouped-matmul-v5-shape-contract",
        }
        report_path = Path(artifacts_dir) / "ops_transformer_grouped_matmul_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    if output is None:
        return {"status": "blocked", "reason": "Baseline benchmark did not produce an output tensor."}

    baseline_tensor = output[0] if isinstance(output, (list, tuple)) else output
    max_abs_diff = (baseline_tensor.float().cpu() - reference).abs().max().item()
    report = {
        "status": "ok",
        "variant": VARIANT.as_dict(),
        "entrypoint": "torch_npu.npu_grouped_matmul",
        "output_type": str(type(output)),
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
        "reference_contract": "bf16_rounded_matmul",
    }
    report_path = Path(artifacts_dir) / "ops_transformer_grouped_matmul_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
