from __future__ import annotations

import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import torch

from pto_kernels.bench.adapter_utils import compile_pto_kernel, describe_pto, temporary_env
from pto_kernels.ops.posembedding.dequant_rope_quant_kvcache.runtime import VARIANTS, make_inputs, run_pto_variant


KERNEL = "python/pto_kernels/ops/posembedding/dequant_rope_quant_kvcache/kernel.py"
META = "python/pto_kernels/ops/posembedding/dequant_rope_quant_kvcache/meta.py"


def describe(repo_root, spec):
    return describe_pto(repo_root, KERNEL, META)


def compile_kernel(repo_root, spec, artifacts_dir):
    return compile_pto_kernel(repo_root, KERNEL, artifacts_dir)


def _variant_env(variant) -> dict[str, str]:
    return {
        "PTO_ROPE_KVCACHE_TOTAL_ROWS": str(variant.batch),
        "PTO_ROPE_KVCACHE_BLOCK_DIM": os.environ.get("PTO_ROPE_KVCACHE_BLOCK_DIM", "4"),
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
        artifact_paths: list[str] = []
        for variant in VARIANTS:
            variant_dir = Path(artifacts_dir) / variant.label
            kernel_file = repo_root / KERNEL
            spec_obj = importlib.util.spec_from_file_location(f"pto_dequant_rope_quant_kvcache_kernel_{variant.label}", kernel_file)
            if spec_obj is None or spec_obj.loader is None:
                raise ImportError(f"Unable to import {kernel_file}")

            with temporary_env(_variant_env(variant)):
                module = importlib.util.module_from_spec(spec_obj)
                spec_obj.loader.exec_module(module)
                wrapper = module.build_jit_wrapper(output_dir=variant_dir)
                build = getattr(wrapper, "_build", None)
                if callable(build):
                    build()

                inputs = make_inputs(variant, device_index=int(spec.device.get("id", 0)))
                reference = inputs["reference"]

                for _ in range(spec.bench.warmup):
                    run_pto_variant(wrapper, inputs)
                torch.npu.synchronize()

                timings_ms = []
                output = None
                for _ in range(spec.bench.repeat):
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    output = run_pto_variant(wrapper, inputs)
                    torch.npu.synchronize()
                    timings_ms.append((time.perf_counter() - start) * 1000.0)

                if output is None:
                    raise RuntimeError(f"PTO benchmark did not produce outputs for {variant.label}.")

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
                artifact_paths.extend([str(path) for path in getattr(wrapper, "_artifact_paths", lambda: ())()])
    except Exception as exc:
        report = {"status": "blocked", "reason": f"PTO execution failed: {exc}", "variants": [variant.as_dict() for variant in VARIANTS]}
        report_path = Path(artifacts_dir) / "ptodsl_dequant_rope_quant_kvcache_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    max_abs_diff = max(max(item["correctness"].values()) for item in variant_reports)
    report = {
        "status": "ok",
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
        "artifact_paths": artifact_paths,
        "reference_contract": "int32_weight_scale_half_rope_quant_kvcache_bsnd_2d",
    }
    report_path = Path(artifacts_dir) / "ptodsl_dequant_rope_quant_kvcache_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
