from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.moe.moe_token_permute_grad.runtime import (
    VARIANT,
    VARIANTS,
    make_token_permute_grad_inputs,
    run_torch_npu_moe_token_permute_grad,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "moe", "moe_token_permute_grad", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_moe_token_permute_grad"
    summary["seed_variant"] = {"default": VARIANT.as_dict(), "variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    del repo_root, spec, artifacts_dir
    if not hasattr(torch_npu, "npu_moe_token_permute_grad"):
        return {
            "status": "blocked",
            "reason": "torch_npu does not expose npu_moe_token_permute_grad on this environment.",
        }
    return {
        "status": "runtime_builtin",
        "entrypoint": "torch_npu.npu_moe_token_permute_grad",
        "note": (
            "Baseline execution uses the installed custom ops runtime package. "
            "The constrained slice fixes probsOptional=None, paddedMode=false, and the current Python-visible host contract to topK=1."
        ),
    }


def benchmark(repo_root, spec, artifacts_dir):
    del repo_root
    try:
        variant_reports = []
        baseline_limitations = []
        for variant in VARIANTS:
            inputs = make_token_permute_grad_inputs(variant, device_index=int(spec.device.get("id", 0)))
            for _ in range(spec.bench.warmup):
                run_torch_npu_moe_token_permute_grad(inputs)
            torch.npu.synchronize()

            timings_ms = []
            output = None
            for _ in range(spec.bench.repeat):
                torch.npu.synchronize()
                start = time.perf_counter()
                output = run_torch_npu_moe_token_permute_grad(inputs)
                torch.npu.synchronize()
                timings_ms.append((time.perf_counter() - start) * 1000.0)

            if output is None:
                raise RuntimeError(f"Baseline benchmark did not produce output tensors for {variant.label}.")

            out_diff = (output.float().cpu() - inputs["reference_out"]).abs().max().item()
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
            "entrypoint": "torch_npu.npu_moe_token_permute_grad",
            "reason": str(exc),
        }
        report_path = Path(artifacts_dir) / "ops_transformer_moe_token_permute_grad_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    max_abs_diff = max(item["correctness"]["max_abs_diff"] for item in variant_reports)
    report = {
        "status": "ok",
        "variants": [item["variant"] for item in variant_reports],
        "entrypoint": "torch_npu.npu_moe_token_permute_grad",
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
        "reference_contract": "top1_sorted_indices_gather_no_probs",
        "baseline_limitations": sorted(set(baseline_limitations)),
    }
    report_path = Path(artifacts_dir) / "ops_transformer_moe_token_permute_grad_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
