from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.attention.attention_update.runtime import (
    VARIANT,
    VARIANTS,
    make_attention_update_inputs,
    run_torch_npu_attention_update,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "attention", "attention_update", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_attention_update"
    summary["seed_variant"] = {"default": VARIANT.as_dict(), "variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    del repo_root, spec, artifacts_dir
    if not hasattr(torch_npu, "npu_attention_update"):
        return {
            "status": "blocked",
            "reason": "torch_npu does not expose npu_attention_update on this environment.",
        }
    return {
        "status": "runtime_builtin",
        "entrypoint": "torch_npu.npu_attention_update",
        "note": "Baseline execution uses the installed runtime package on the constrained sp=2, updateType=0 slice.",
    }


def benchmark(repo_root, spec, artifacts_dir):
    del repo_root
    try:
        variant_reports = []
        for variant in VARIANTS:
            inputs = make_attention_update_inputs(variant, device_index=int(spec.device.get("id", 0)))
            for _ in range(spec.bench.warmup):
                run_torch_npu_attention_update(inputs)
            torch.npu.synchronize()

            timings_ms = []
            output = None
            for _ in range(spec.bench.repeat):
                torch.npu.synchronize()
                start = time.perf_counter()
                output = run_torch_npu_attention_update(inputs)
                torch.npu.synchronize()
                timings_ms.append((time.perf_counter() - start) * 1000.0)

            if output is None:
                raise RuntimeError(f"Baseline benchmark did not produce output tensors for {variant.label}.")

            out, lse_out = output
            if lse_out is not None:
                raise RuntimeError("Constrained baseline slice expects updateType=0 and no lseOut tensor.")
            out_diff = (out.float().cpu() - inputs["reference"]).abs().max().item()
            variant_reports.append(
                {
                    "variant": variant.as_dict(),
                    "shape_summary": variant.shape_summary,
                    "timings_ms": {
                        "median": statistics.median(timings_ms),
                        "min": min(timings_ms),
                        "max": max(timings_ms),
                    },
                    "correctness": {
                        "out_max_abs_diff": out_diff,
                        "max_abs_diff": out_diff,
                        "passes": bool(out_diff <= spec.correctness.atol),
                    },
                }
            )
    except Exception as exc:
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "entrypoint": "torch_npu.npu_attention_update",
            "reason": str(exc),
        }
        report_path = Path(artifacts_dir) / "ops_transformer_attention_update_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    max_abs_diff = max(item["correctness"]["max_abs_diff"] for item in variant_reports)
    report = {
        "status": "ok",
        "variants": [item["variant"] for item in variant_reports],
        "entrypoint": "torch_npu.npu_attention_update",
        "shape_summaries": [item["shape_summary"] for item in variant_reports],
        "timings_ms": {
            "median": max(item["timings_ms"]["median"] for item in variant_reports),
            "min": min(item["timings_ms"]["min"] for item in variant_reports),
            "max": max(item["timings_ms"]["max"] for item in variant_reports),
        },
        "correctness": {
            "max_abs_diff": max_abs_diff,
            "atol": spec.correctness.atol,
            "rtol": spec.correctness.rtol,
            "passes": bool(all(item["correctness"]["passes"] for item in variant_reports)),
        },
        "reference_contract": "sp2_update_type0",
        "variant_reports": variant_reports,
    }
    report_path = Path(artifacts_dir) / "ops_transformer_attention_update_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
