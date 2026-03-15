from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.moe.moe_gating_top_k.runtime import (
    VARIANT,
    VARIANTS,
    make_moe_gating_top_k_inputs,
    run_torch_npu_moe_gating_top_k,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "moe", "moe_gating_top_k", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_moe_gating_top_k"
    summary["seed_variant"] = {"default": VARIANT.as_dict(), "variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    del repo_root, spec, artifacts_dir
    if not hasattr(torch_npu, "npu_moe_gating_top_k"):
        return {
            "status": "blocked",
            "reason": "torch_npu does not expose npu_moe_gating_top_k on this environment.",
        }
    return {
        "status": "runtime_builtin",
        "entrypoint": "torch_npu.npu_moe_gating_top_k",
        "note": (
            "Baseline execution uses the installed custom ops runtime package. "
            "The constrained slice fixes topK=1, groupCount=1, biasOptional=None, "
            "normType=sigmoid, and outFlag=false on 2D input."
        ),
    }


def benchmark(repo_root, spec, artifacts_dir):
    del repo_root
    try:
        variant_reports = []
        for variant in VARIANTS:
            inputs = make_moe_gating_top_k_inputs(variant, device_index=int(spec.device.get("id", 0)))
            for _ in range(spec.bench.warmup):
                run_torch_npu_moe_gating_top_k(inputs)
            torch.npu.synchronize()

            timings_ms = []
            output = None
            for _ in range(spec.bench.repeat):
                torch.npu.synchronize()
                start = time.perf_counter()
                output = run_torch_npu_moe_gating_top_k(inputs)
                torch.npu.synchronize()
                timings_ms.append((time.perf_counter() - start) * 1000.0)

            if output is None:
                raise RuntimeError(f"Baseline benchmark did not produce output tensors for {variant.label}.")

            y_out, expert_idx_out, norm_out = output
            y_diff = (y_out.float().cpu() - inputs["reference_y_out"]).abs().max().item()
            expert_diff = (expert_idx_out.to(torch.int32).cpu() - inputs["reference_expert_idx_out"]).abs().max().item()
            norm_diff = (norm_out.float().cpu() - inputs["reference_norm_out"]).abs().max().item()
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
                        "y_max_abs_diff": y_diff,
                        "expert_idx_max_abs_diff": float(expert_diff),
                        "norm_out_max_abs_diff": norm_diff,
                        "max_abs_diff": max(y_diff, float(expert_diff), norm_diff),
                        "passes": bool(
                            y_diff <= spec.correctness.atol and expert_diff == 0 and norm_diff <= spec.correctness.atol
                        ),
                    },
                }
            )
    except Exception as exc:
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "entrypoint": "torch_npu.npu_moe_gating_top_k",
            "reason": str(exc),
        }
        report_path = Path(artifacts_dir) / "ops_transformer_moe_gating_top_k_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    max_abs_diff = max(item["correctness"]["max_abs_diff"] for item in variant_reports)
    report = {
        "status": "ok",
        "variants": [item["variant"] for item in variant_reports],
        "entrypoint": "torch_npu.npu_moe_gating_top_k",
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
        "variant_reports": variant_reports,
        "reference_contract": "top1_sigmoid_no_group_no_bias_outflag_false",
    }
    report_path = Path(artifacts_dir) / "ops_transformer_moe_gating_top_k_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
