from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import torch

from pto_kernels.bench.adapter_utils import compile_pto_kernel, describe_pto, load_module, temporary_env
from pto_kernels.ops.attention.recurrent_gated_delta_rule.runtime import (
    VARIANTS,
    make_recurrent_gated_delta_rule_inputs,
    run_pto_recurrent_gated_delta_rule_variant,
)


KERNEL = "python/pto_kernels/ops/attention/recurrent_gated_delta_rule/kernel.py"
META = "python/pto_kernels/ops/attention/recurrent_gated_delta_rule/meta.py"


def describe(repo_root, spec):
    return describe_pto(repo_root, KERNEL, META)


def compile_kernel(repo_root, spec, artifacts_dir):
    return compile_pto_kernel(repo_root, KERNEL, artifacts_dir)


def _variant_env(variant) -> dict[str, str]:
    rows = variant.total_tokens * variant.num_value_heads
    return {
        "PTO_RECURRENT_GATED_DELTA_RULE_ROWS": str(rows),
        "PTO_RECURRENT_GATED_DELTA_RULE_DIM": str(variant.dim),
        "PTO_RECURRENT_GATED_DELTA_RULE_BLOCK_DIM": os.environ.get("PTO_RECURRENT_GATED_DELTA_RULE_BLOCK_DIM", "8"),
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

                inputs = make_recurrent_gated_delta_rule_inputs(variant, device_index=int(spec.device.get("id", 0)))

                for _ in range(spec.bench.warmup):
                    run_pto_recurrent_gated_delta_rule_variant(wrapper, inputs)
                torch.npu.synchronize()

                timings_ms = []
                output = None
                final_state = None
                for _ in range(spec.bench.repeat):
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    output, final_state = run_pto_recurrent_gated_delta_rule_variant(wrapper, inputs)
                    torch.npu.synchronize()
                    timings_ms.append((time.perf_counter() - start) * 1000.0)

                if output is None or final_state is None:
                    raise RuntimeError(f"PTO benchmark did not produce output tensors for {variant.label}.")

                out_diff = (output.float().cpu() - inputs["reference_out"].float()).abs().max().item()
                state_diff = (final_state.float().cpu() - inputs["reference_state"].float()).abs().max().item()
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
                            "state_max_abs_diff": state_diff,
                            "max_abs_diff": max(out_diff, state_diff),
                            "passes": bool(
                                out_diff <= spec.correctness.atol and state_diff <= spec.correctness.atol
                            ),
                        },
                    }
                )
                artifact_paths.extend([str(path) for path in getattr(wrapper, "_artifact_paths", lambda: ())()])
    except Exception as exc:
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "reason": f"PTO compile or execution failed: {exc}",
        }
        report_path = Path(artifacts_dir) / "ptodsl_recurrent_gated_delta_rule_benchmark.json"
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
            "passes": bool(all(item["correctness"]["passes"] for item in variant_reports)),
        },
        "reference_contract": "nd_recurrent_no_gk_identity_state_slots",
        "variant_reports": variant_reports,
        "artifact_paths": artifact_paths,
    }
    report_path = Path(artifacts_dir) / "ptodsl_recurrent_gated_delta_rule_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
