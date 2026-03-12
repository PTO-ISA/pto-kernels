from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.ffn.ffn.runtime import (
    VARIANT,
    make_dense_relu_inputs,
    run_torch_npu_ffn,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "ffn", "ffn", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_ffn"
    summary["seed_variant"] = VARIANT.as_dict()
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    if not hasattr(torch_npu, "npu_ffn"):
        return {
            "status": "blocked",
            "reason": "torch_npu does not expose npu_ffn on this environment.",
        }
    return {
        "status": "runtime_builtin",
        "entrypoint": "torch_npu.npu_ffn",
        "note": (
            "Baseline execution relies on the installed custom ops runtime package. "
            "The seed variant is constrained to dense float16 FFN with relu activation and no bias."
        ),
    }


def benchmark(repo_root, spec, artifacts_dir):
    inputs = make_dense_relu_inputs(device_index=int(spec.device.get("id", 0)))
    reference = inputs["reference"]
    try:
        for _ in range(spec.bench.warmup):
            run_torch_npu_ffn(inputs)
        torch.npu.synchronize()

        timings_ms = []
        output = None
        for _ in range(spec.bench.repeat):
            torch.npu.synchronize()
            start = time.perf_counter()
            output = run_torch_npu_ffn(inputs)
            torch.npu.synchronize()
            timings_ms.append((time.perf_counter() - start) * 1000.0)
    except Exception as exc:  # pragma: no cover - exercised on NPU bring-up hosts
        report = {
            "status": "blocked",
            "variant": VARIANT.as_dict(),
            "entrypoint": "torch_npu.npu_ffn",
            "reason": str(exc),
        }
        report_path = Path(artifacts_dir) / "ops_transformer_ffn_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    if output is None:
        return {"status": "blocked", "reason": "Baseline benchmark did not produce an output tensor."}

    max_abs_diff = (output.float().cpu() - reference).abs().max().item()
    report = {
        "status": "ok",
        "variant": VARIANT.as_dict(),
        "entrypoint": "torch_npu.npu_ffn",
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
        "reference_contract": "fp16_dense_relu_ffn_torch_ops",
    }
    report_path = Path(artifacts_dir) / "ops_transformer_ffn_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
