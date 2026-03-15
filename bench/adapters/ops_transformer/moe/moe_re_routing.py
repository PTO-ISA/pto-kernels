from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.moe.moe_re_routing.runtime import (
    VARIANT,
    VARIANTS,
    make_moe_re_routing_inputs,
    run_torch_npu_moe_re_routing,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "moe", "moe_re_routing", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_moe_re_routing"
    summary["seed_variant"] = {"default": VARIANT.as_dict(), "variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    del repo_root, spec, artifacts_dir
    if not hasattr(torch_npu, "npu_moe_re_routing"):
        return {
            "status": "blocked",
            "reason": "torch_npu does not expose npu_moe_re_routing on this environment.",
        }
    return {
        "status": "runtime_builtin",
        "entrypoint": "torch_npu.npu_moe_re_routing",
        "note": (
            "Baseline execution uses the installed custom ops runtime package. "
            "The constrained slice fixes per_token_scales enabled, expert_token_num_type=1, and idx_type=0."
        ),
    }


def benchmark(repo_root, spec, artifacts_dir):
    del repo_root
    try:
        variant_reports = []
        for variant in VARIANTS:
            inputs = make_moe_re_routing_inputs(variant, device_index=int(spec.device.get("id", 0)))
            for _ in range(spec.bench.warmup):
                run_torch_npu_moe_re_routing(inputs)
            torch.npu.synchronize()

            timings_ms = []
            output = None
            for _ in range(spec.bench.repeat):
                torch.npu.synchronize()
                start = time.perf_counter()
                output = run_torch_npu_moe_re_routing(inputs)
                torch.npu.synchronize()
                timings_ms.append((time.perf_counter() - start) * 1000.0)

            if output is None:
                raise RuntimeError(f"Baseline benchmark did not produce output tensors for {variant.label}.")

            permute_tokens, permute_scales, permute_idx, expert_token_num = output
            token_diff = (permute_tokens.float().cpu() - inputs["reference_permute_tokens"]).abs().max().item()
            scale_diff = (permute_scales.float().cpu() - inputs["reference_permute_per_token_scales"]).abs().max().item()
            idx_diff = (
                permute_idx.to(torch.int32).cpu() - inputs["reference_permute_token_idx"].to(torch.int32)
            ).abs().max().item()
            expert_diff = (
                expert_token_num.to(torch.int32).cpu() - inputs["reference_expert_token_num"].to(torch.int32)
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
                        "token_max_abs_diff": token_diff,
                        "scale_max_abs_diff": scale_diff,
                        "idx_max_abs_diff": idx_diff,
                        "expert_token_num_max_abs_diff": expert_diff,
                        "max_abs_diff": max(token_diff, scale_diff, idx_diff, expert_diff),
                    },
                }
            )
    except Exception as exc:
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "entrypoint": "torch_npu.npu_moe_re_routing",
            "reason": str(exc),
        }
        report_path = Path(artifacts_dir) / "ops_transformer_moe_re_routing_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    max_abs_diff = max(item["correctness"]["max_abs_diff"] for item in variant_reports)
    report = {
        "status": "ok",
        "variants": [item["variant"] for item in variant_reports],
        "entrypoint": "torch_npu.npu_moe_re_routing",
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
            "passes": bool(all(item["correctness"]["max_abs_diff"] <= spec.correctness.atol for item in variant_reports)),
        },
        "variant_reports": variant_reports,
        "reference_contract": "fp16_moe_re_routing_idx0_with_scales",
    }
    report_path = Path(artifacts_dir) / "ops_transformer_moe_re_routing_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
