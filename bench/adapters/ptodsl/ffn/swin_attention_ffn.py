from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import torch

from pto_kernels.bench.adapter_utils import compile_pto_kernel, describe_pto, load_module, temporary_env
from pto_kernels.ops.ffn.swin_attention_ffn.runtime import VARIANTS, make_inputs, run_pto_variant


KERNEL = "python/pto_kernels/ops/ffn/swin_attention_ffn/kernel.py"
META = "python/pto_kernels/ops/ffn/swin_attention_ffn/meta.py"


def describe(repo_root, spec):
    return describe_pto(repo_root, KERNEL, META)


def compile_kernel(repo_root, spec, artifacts_dir):
    return compile_pto_kernel(repo_root, KERNEL, artifacts_dir)


def _variant_env(variant) -> dict[str, str]:
    return {
        "PTO_SWIN_ATTENTION_FFN_TOKENS": str(variant.batch * variant.seq),
        "PTO_SWIN_ATTENTION_FFN_HIDDEN": str(variant.hidden),
        "PTO_SWIN_ATTENTION_FFN_BASE_M": os.environ.get("PTO_SWIN_ATTENTION_FFN_BASE_M", "128"),
        "PTO_SWIN_ATTENTION_FFN_BASE_N": os.environ.get("PTO_SWIN_ATTENTION_FFN_BASE_N", "128"),
        "PTO_SWIN_ATTENTION_FFN_BASE_K": os.environ.get("PTO_SWIN_ATTENTION_FFN_BASE_K", "128"),
        "PTO_SWIN_ATTENTION_FFN_BLOCK_DIM": os.environ.get("PTO_SWIN_ATTENTION_FFN_BLOCK_DIM", "24"),
        "PTO_SWIN_ATTENTION_FFN_ADD_BLOCK_DIM": os.environ.get("PTO_SWIN_ATTENTION_FFN_ADD_BLOCK_DIM", "24"),
    }


def benchmark(repo_root, spec, artifacts_dir):
    try:
        variant_reports = []
        artifact_paths: list[str] = []
        kernel_file = repo_root / KERNEL
        for variant in VARIANTS:
            with temporary_env(_variant_env(variant)):
                module = load_module(Path(kernel_file))
                builder = getattr(module, "build_jit_wrapper", None)
                if not callable(builder):
                    return {"status": "blocked", "reason": "kernel module does not expose build_jit_wrapper(output_dir)"}

                wrapper = builder(output_dir=Path(artifacts_dir) / variant.label)
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
                    raise RuntimeError(f"PTO benchmark did not produce an output tensor for {variant.label}.")

                max_abs_diff = (output.float().cpu() - reference).abs().max().item()
                variant_reports.append(
                    {
                        "variant": variant.as_dict(),
                        "shape_summary": variant.shape_summary,
                        "timings_ms": {
                            "median": statistics.median(timings_ms),
                            "min": min(timings_ms),
                            "max": max(timings_ms),
                        },
                        "correctness": {"max_abs_diff": max_abs_diff},
                    }
                )
                artifact_paths.extend([str(path) for path in getattr(wrapper, "_artifact_paths", lambda: ())()])
    except Exception as exc:
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "reason": f"PTO compile failed: {exc}",
        }
        report_path = Path(artifacts_dir) / "ptodsl_swin_attention_ffn_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    max_abs_diff = max(item["correctness"]["max_abs_diff"] for item in variant_reports)
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
        "block_utilization": {
            "nominal_tokens": VARIANTS[1].tokens,
            "nominal_base_m": 128,
            "nominal_tiles": VARIANTS[1].tokens // 128,
            "requested_block_dim": 24,
            "uses_all_blocks": True,
        },
        "reference_contract": "fp16_zero_shift_swin_attention_ffn",
    }
    report_path = Path(artifacts_dir) / "ptodsl_swin_attention_ffn_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
