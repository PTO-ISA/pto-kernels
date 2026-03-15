from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.attention.recurrent_gated_delta_rule.runtime import (
    VARIANTS,
    make_recurrent_gated_delta_rule_inputs,
    run_torch_npu_recurrent_gated_delta_rule,
    run_torch_npu_recurrent_gated_delta_rule_functional,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "attention", "recurrent_gated_delta_rule", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_recurrent_gated_delta_rule"
    summary["seed_variant"] = {
        "default": VARIANTS[0].as_dict(),
        "variants": [variant.as_dict() for variant in VARIANTS],
    }
    return summary


def benchmark(repo_root, spec, artifacts_dir):
    if not hasattr(torch_npu, "npu_recurrent_gated_delta_rule_functional"):
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "reason": "torch_npu does not expose npu_recurrent_gated_delta_rule_functional on this environment.",
            "entrypoint": "torch_npu.npu_recurrent_gated_delta_rule_functional",
        }
        report_path = Path(artifacts_dir) / "ops_transformer_recurrent_gated_delta_rule_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    try:
        variant_reports = []
        for variant in VARIANTS:
            inputs = make_recurrent_gated_delta_rule_inputs(variant, device_index=int(spec.device.get("id", 0)))
            for _ in range(spec.bench.warmup):
                run_torch_npu_recurrent_gated_delta_rule_functional(inputs)
            torch.npu.synchronize()

            timings_ms = []
            output = None
            final_state = None
            for _ in range(spec.bench.repeat):
                inputs["state_functional"].copy_(inputs["state"])
                torch.npu.synchronize()
                start = time.perf_counter()
                output, final_state = run_torch_npu_recurrent_gated_delta_rule_functional(inputs)
                torch.npu.synchronize()
                timings_ms.append((time.perf_counter() - start) * 1000.0)

            if output is None or final_state is None:
                raise RuntimeError(f"Baseline benchmark did not produce output tensors for {variant.label}.")

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
                        "max_abs_diff": out_diff,
                        "passes": bool(out_diff <= spec.correctness.atol),
                    },
                }
            )
    except Exception as exc:
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "reason": f"Baseline execution failed: {exc}",
            "entrypoint": "torch_npu.npu_recurrent_gated_delta_rule_functional",
        }
        report_path = Path(artifacts_dir) / "ops_transformer_recurrent_gated_delta_rule_benchmark.json"
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
        "entrypoint": "torch_npu.npu_recurrent_gated_delta_rule_functional",
        "reference_contract": "nd_recurrent_no_gk",
        "variant_reports": variant_reports,
        "baseline_limitations": [
            "baseline timing uses the functional entrypoint to observe both output and final_state",
            "baseline correctness is validated on output only; final_state semantics remain a host-contract gap for this first bounded slice",
        ],
    }
    report_path = Path(artifacts_dir) / "ops_transformer_recurrent_gated_delta_rule_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
