from __future__ import annotations

import json
from pathlib import Path

from pto_kernels.bench.adapter_utils import compile_pto_kernel, describe_pto, load_module
from pto_kernels.ops.mc2.matmul_reduce_scatter.runtime import (
    VARIANT,
    run_distributed_pto_benchmark,
)


KERNEL = "python/pto_kernels/ops/mc2/matmul_reduce_scatter/kernel.py"
META = "python/pto_kernels/ops/mc2/matmul_reduce_scatter/meta.py"


def describe(repo_root, spec):
    return describe_pto(repo_root, KERNEL, META)


def compile_kernel(repo_root, spec, artifacts_dir):
    return compile_pto_kernel(repo_root, KERNEL, artifacts_dir)


def benchmark(repo_root, spec, artifacts_dir):
    kernel_file = repo_root / KERNEL
    module = load_module(Path(kernel_file))
    builder = getattr(module, "build_jit_wrapper", None)
    if not callable(builder):
        return {"status": "blocked", "reason": "kernel module does not expose build_jit_wrapper(output_dir)"}

    wrapper = builder(output_dir=artifacts_dir / "compile_probe")
    build = getattr(wrapper, "_build", None)
    try:
        if callable(build):
            build()
    except Exception as exc:  # pragma: no cover - exercised on NPU bring-up hosts
        report = {"status": "blocked", "variant": VARIANT.as_dict(), "reason": f"PTO compile failed: {exc}"}
        report_path = Path(artifacts_dir) / "ptodsl_matmul_reduce_scatter_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    report = run_distributed_pto_benchmark(
        artifacts_dir=Path(artifacts_dir),
        warmup=spec.bench.warmup,
        repeat=spec.bench.repeat,
    )
    if report.get("status") == "ok":
        max_abs_diff = float(report["correctness"]["max_abs_diff"])
        report["correctness"].update(
            {
                "atol": spec.correctness.atol,
                "rtol": spec.correctness.rtol,
                "passes": bool(max_abs_diff <= spec.correctness.atol),
            }
        )
        report["artifact_paths"] = [str(path) for path in getattr(wrapper, "_artifact_paths", lambda: ())()]

    report_path = Path(artifacts_dir) / "ptodsl_matmul_reduce_scatter_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
