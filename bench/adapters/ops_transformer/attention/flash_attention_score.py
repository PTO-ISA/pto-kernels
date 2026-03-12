from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.attention.flash_attention_score.runtime import (
    VARIANT,
    make_dense_bnsd_inputs,
    run_torch_npu_flash_attention_score,
)


def describe(repo_root, spec):
    summary = describe_baseline(
        repo_root,
        "attention",
        "flash_attention_score",
        spec.inventory_ref,
    )
    summary["runtime_entrypoint"] = "torch_npu.npu_fusion_attention_v2"
    summary["seed_variant"] = VARIANT.as_dict()
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    if not hasattr(torch_npu, "npu_fusion_attention_v2"):
        return {
            "status": "blocked",
            "reason": "torch_npu does not expose npu_fusion_attention_v2 on this environment.",
        }
    return {
        "status": "runtime_builtin",
        "entrypoint": "torch_npu.npu_fusion_attention_v2",
        "note": (
            "Baseline execution relies on the installed custom ops runtime package. "
            "The seed variant is constrained to dense BNSD attention with no masks, no dropout, and fp16 inputs."
        ),
    }


def benchmark(repo_root, spec, artifacts_dir):
    inputs = make_dense_bnsd_inputs(device_index=int(spec.device.get("id", 0)))
    reference = inputs["reference"]
    try:
        for _ in range(spec.bench.warmup):
            run_torch_npu_flash_attention_score(inputs)
        torch.npu.synchronize()

        timings_ms = []
        output = None
        for _ in range(spec.bench.repeat):
            torch.npu.synchronize()
            start = time.perf_counter()
            output = run_torch_npu_flash_attention_score(inputs)
            torch.npu.synchronize()
            timings_ms.append((time.perf_counter() - start) * 1000.0)
    except Exception as exc:  # pragma: no cover - exercised on NPU bring-up hosts
        report = {
            "status": "blocked",
            "variant": VARIANT.as_dict(),
            "entrypoint": "torch_npu.npu_fusion_attention_v2",
            "reason": str(exc),
        }
        report_path = Path(artifacts_dir) / "ops_transformer_flash_attention_score_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    if output is None:
        return {"status": "blocked", "reason": "Baseline benchmark did not produce an output tensor."}

    attention_out = output[0]
    max_abs_diff = (attention_out.float().cpu() - reference).abs().max().item()
    report = {
        "status": "ok",
        "variant": VARIANT.as_dict(),
        "entrypoint": "torch_npu.npu_fusion_attention_v2",
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
        "reference_contract": "fp16_bnsd_scaled_dot_product_attention",
    }
    report_path = Path(artifacts_dir) / "ops_transformer_flash_attention_score_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
