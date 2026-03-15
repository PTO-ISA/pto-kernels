from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.posembedding.dequant_rope_quant_kvcache.runtime import (
    VARIANTS,
    make_inputs,
    run_torch_npu_dequant_rope_quant_kvcache,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "posembedding", "dequant_rope_quant_kvcache", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_dequant_rope_quant_kvcache"
    summary["seed_variant"] = {"variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_baseline(repo_root, spec):
    if not hasattr(torch_npu, "npu_dequant_rope_quant_kvcache"):
        return {"status": "blocked", "reason": "torch_npu does not expose npu_dequant_rope_quant_kvcache on this environment."}
    return {
        "status": "runtime_builtin",
        "entrypoint": "torch_npu.npu_dequant_rope_quant_kvcache",
        "note": "Baseline execution relies on the installed runtime builtin for the constrained int32 dequant_rope_quant_kvcache seed.",
    }


def _correctness(output_tuple, reference):
    q_out, k_out, v_out, k_cache_out, v_cache_out = output_tuple
    return {
        "q_max_abs_diff": (q_out.float().cpu() - reference["q"]).abs().max().item(),
        "k_max_abs_diff": (k_out.float().cpu() - reference["k"]).abs().max().item(),
        "v_max_abs_diff": (v_out.float().cpu() - reference["v"]).abs().max().item(),
        "k_cache_max_abs_diff": (k_cache_out.cpu().float() - reference["k_cache"].float()).abs().max().item(),
        "v_cache_max_abs_diff": (v_cache_out.cpu().float() - reference["v_cache"].float()).abs().max().item(),
    }


def benchmark(repo_root, spec, artifacts_dir):
    try:
        variant_reports = []
        for variant in VARIANTS:
            inputs = make_inputs(variant, device_index=int(spec.device.get("id", 0)))
            reference = inputs["reference"]

            for _ in range(spec.bench.warmup):
                run_torch_npu_dequant_rope_quant_kvcache(inputs)
            torch.npu.synchronize()

            timings_ms = []
            output = None
            for _ in range(spec.bench.repeat):
                torch.npu.synchronize()
                start = time.perf_counter()
                output = run_torch_npu_dequant_rope_quant_kvcache(inputs)
                torch.npu.synchronize()
                timings_ms.append((time.perf_counter() - start) * 1000.0)

            if output is None:
                raise RuntimeError(f"Baseline did not produce outputs for {variant.label}.")

            correctness = _correctness(output, reference)
            variant_reports.append(
                {
                    "variant": variant.as_dict(),
                    "shape_summary": inputs["shape_summary"],
                    "timings_ms": {
                        "median": statistics.median(timings_ms),
                        "min": min(timings_ms),
                        "max": max(timings_ms),
                    },
                    "correctness": correctness,
                }
            )
    except Exception as exc:
        report = {"status": "blocked", "reason": f"Baseline execution failed: {exc}", "entrypoint": "torch_npu.npu_dequant_rope_quant_kvcache"}
        report_path = Path(artifacts_dir) / "ops_transformer_dequant_rope_quant_kvcache_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    max_abs_diff = max(max(item["correctness"].values()) for item in variant_reports)
    report = {
        "status": "ok",
        "entrypoint": "torch_npu.npu_dequant_rope_quant_kvcache",
        "variants": [item["variant"] for item in variant_reports],
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
            "passes": bool(max_abs_diff <= spec.correctness.atol),
        },
        "variant_reports": variant_reports,
        "reference_contract": "int32_weight_scale_half_rope_quant_kvcache_bsnd_2d",
    }
    report_path = Path(artifacts_dir) / "ops_transformer_dequant_rope_quant_kvcache_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
