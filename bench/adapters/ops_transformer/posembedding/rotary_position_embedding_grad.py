from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.posembedding.rotary_position_embedding_grad.runtime import (
    VARIANTS,
    make_inputs,
    run_torch_npu_rotary_position_embedding_grad,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "posembedding", "rotary_position_embedding_grad", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_rotary_mul_backward"
    summary["seed_variant"] = {"variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    if not hasattr(torch_npu, "npu_rotary_mul_backward"):
        return {
            "status": "blocked",
            "reason": "torch_npu does not expose npu_rotary_mul_backward on this environment.",
        }
    return {
        "status": "runtime_builtin",
        "entrypoint": "torch_npu.npu_rotary_mul_backward",
        "note": "Baseline execution uses the installed half-mode rotary backward runtime path exposed through torch_npu.npu_rotary_mul_backward.",
    }


def benchmark(repo_root, spec, artifacts_dir):
    variant_reports = []
    try:
        for variant in VARIANTS:
            inputs = make_inputs(variant, device_index=int(spec.device.get("id", 0)))

            for _ in range(spec.bench.warmup):
                run_torch_npu_rotary_position_embedding_grad(inputs)
            torch.npu.synchronize()

            timings_ms = []
            output = None
            for _ in range(spec.bench.repeat):
                torch.npu.synchronize()
                start = time.perf_counter()
                output = run_torch_npu_rotary_position_embedding_grad(inputs)
                torch.npu.synchronize()
                timings_ms.append((time.perf_counter() - start) * 1000.0)

            if output is None:
                raise RuntimeError(f"Baseline benchmark did not produce outputs for {variant.label}.")

            dx, dcos, dsin = output
            dx_max_abs_diff = (dx.float().cpu() - inputs["reference_dx"]).abs().max().item()
            dcos_max_abs_diff = (dcos.float().cpu() - inputs["reference_dcos"]).abs().max().item()
            dsin_max_abs_diff = (dsin.float().cpu() - inputs["reference_dsin"]).abs().max().item()
            dcos_runtime_zero = bool(torch.count_nonzero(dcos).item() == 0)
            dsin_runtime_zero = bool(torch.count_nonzero(dsin).item() == 0)
            missing_optional_grads = dcos_runtime_zero and dsin_runtime_zero
            correctness = {
                "dx_max_abs_diff": dx_max_abs_diff,
                "dcos_max_abs_diff": dcos_max_abs_diff,
                "dsin_max_abs_diff": dsin_max_abs_diff,
                "supported_outputs": ["dx"],
                "unsupported_outputs": ["dcos", "dsin"] if missing_optional_grads else [],
                "passes": bool(dx_max_abs_diff <= spec.correctness.atol),
            }
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
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "entrypoint": "torch_npu.npu_rotary_mul_backward",
            "reason": str(exc),
        }
        report_path = Path(artifacts_dir) / "ops_transformer_rotary_position_embedding_grad_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    max_abs_diff = max(item["correctness"]["dx_max_abs_diff"] for item in variant_reports)
    report = {
        "status": "ok",
        "variants": [item["variant"] for item in variant_reports],
        "entrypoint": "torch_npu.npu_rotary_mul_backward",
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
        "reference_contract": "fp16_half_rotary_grad_bsnd_bnsd_d128_dx_only_baseline",
        "baseline_limitations": [
            "On this host torch_npu.npu_rotary_mul_backward returns zero-filled dcos/dsin outputs.",
            "Baseline correctness and parity are therefore validated on dx only; PTO still validates dx/dcos/dsin against the PTO reference.",
        ],
    }
    report_path = Path(artifacts_dir) / "ops_transformer_rotary_position_embedding_grad_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
