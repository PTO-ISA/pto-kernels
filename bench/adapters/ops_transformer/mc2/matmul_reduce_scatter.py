from __future__ import annotations

import json
from pathlib import Path

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.mc2.matmul_reduce_scatter.runtime import (
    VARIANT,
    baseline_blocker,
    run_distributed_baseline_benchmark,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "mc2", "matmul_reduce_scatter", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_mm_reduce_scatter_base"
    summary["seed_variant"] = VARIANT.as_dict()
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    report = baseline_blocker(device_index=int(spec.device.get("id", 0)))
    environment = report.get("environment", {})
    if report["status"] == "blocked" and environment.get("symbol_available"):
        return {
            "status": "runtime_builtin_distributed",
            "entrypoint": "torch_npu.npu_mm_reduce_scatter_base",
            "note": (
                "Baseline execution uses a local multi-rank HCCL launch from the benchmark adapter. "
                "WORLD_SIZE is provided by the adapter unless PTO_MC2_WORLD_SIZE overrides it."
            ),
            "environment": environment,
        }
    return report


def benchmark(repo_root, spec, artifacts_dir):
    try:
        report = run_distributed_baseline_benchmark(
            artifacts_dir=Path(artifacts_dir),
            warmup=spec.bench.warmup,
            repeat=spec.bench.repeat,
        )
    except Exception as exc:  # pragma: no cover - exercised on NPU bring-up hosts
        report = baseline_blocker(device_index=int(spec.device.get("id", 0)))
        report["reason"] = f"Distributed MC2 baseline launch failed: {exc}"
    report["entrypoint"] = "torch_npu.npu_mm_reduce_scatter_base"
    if report.get("status") == "ok":
        max_abs_diff = float(report["correctness"]["max_abs_diff"])
        report["correctness"].update(
            {
                "atol": spec.correctness.atol,
                "rtol": spec.correctness.rtol,
                "passes": bool(max_abs_diff <= spec.correctness.atol),
            }
        )
    report_path = Path(artifacts_dir) / "ops_transformer_matmul_reduce_scatter_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
