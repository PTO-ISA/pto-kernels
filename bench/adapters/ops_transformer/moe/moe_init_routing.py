from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.moe.moe_init_routing.runtime import (
    VARIANT,
    VARIANTS,
    make_moe_init_routing_inputs,
    run_torch_npu_moe_init_routing,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "moe", "moe_init_routing", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch.ops.npu.npu_moe_init_routing"
    summary["seed_variant"] = {"default": VARIANT.as_dict(), "variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    del repo_root, spec, artifacts_dir
    return {
        "status": "runtime_builtin",
        "entrypoint": "torch.ops.npu.npu_moe_init_routing",
        "note": (
            "Baseline execution uses the installed runtime builtin. "
            "The constrained slice fixes topK=1 and pre-groups expert_idx by expert."
        ),
    }


def benchmark(repo_root, spec, artifacts_dir):
    del repo_root
    try:
        variant_reports = []
        for variant in VARIANTS:
            inputs = make_moe_init_routing_inputs(variant, device_index=int(spec.device.get("id", 0)))
            for _ in range(spec.bench.warmup):
                run_torch_npu_moe_init_routing(inputs)
            torch.npu.synchronize()

            timings_ms = []
            output = None
            for _ in range(spec.bench.repeat):
                torch.npu.synchronize()
                start = time.perf_counter()
                output = run_torch_npu_moe_init_routing(inputs)
                torch.npu.synchronize()
                timings_ms.append((time.perf_counter() - start) * 1000.0)

            if output is None:
                raise RuntimeError(f"Baseline benchmark did not produce output tensors for {variant.label}.")

            expanded_x, expanded_row_idx, expanded_expert_idx = output
            x_diff = (expanded_x.float().cpu() - inputs["reference_expanded_x"]).abs().max().item()
            row_diff = (expanded_row_idx.to(torch.int32).cpu() - inputs["reference_expanded_row_idx"]).abs().max().item()
            expert_diff = (
                expanded_expert_idx.to(torch.int32).cpu() - inputs["reference_expanded_expert_idx"]
            ).abs().max().item()
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
                        "expanded_x_max_abs_diff": x_diff,
                        "expanded_row_idx_max_abs_diff": row_diff,
                        "expanded_expert_idx_max_abs_diff": expert_diff,
                        "max_abs_diff": max(x_diff, float(row_diff), float(expert_diff)),
                    },
                }
            )
    except Exception as exc:
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "entrypoint": "torch.ops.npu.npu_moe_init_routing",
            "reason": str(exc),
        }
        report_path = Path(artifacts_dir) / "ops_transformer_moe_init_routing_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    max_abs_diff = max(item["correctness"]["max_abs_diff"] for item in variant_reports)
    report = {
        "status": "ok",
        "variants": [item["variant"] for item in variant_reports],
        "entrypoint": "torch.ops.npu.npu_moe_init_routing",
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
            "passes": bool(
                all(
                    item["correctness"]["expanded_x_max_abs_diff"] <= spec.correctness.atol
                    and item["correctness"]["expanded_row_idx_max_abs_diff"] == 0
                    and item["correctness"]["expanded_expert_idx_max_abs_diff"] == 0
                    for item in variant_reports
                )
            ),
        },
        "variant_reports": variant_reports,
        "reference_contract": "top1_grouped_expert_idx_init_routing",
    }
    report_path = Path(artifacts_dir) / "ops_transformer_moe_init_routing_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
