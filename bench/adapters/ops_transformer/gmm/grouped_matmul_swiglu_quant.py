from __future__ import annotations

import json
from pathlib import Path
import time

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.gmm.grouped_matmul_swiglu_quant.runtime import (
    VARIANT,
    VARIANTS,
    baseline_blocker,
    probe_baseline_contract,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "gmm", "grouped_matmul_swiglu_quant", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_grouped_matmul_swiglu_quant"
    summary["seed_variant"] = {"default": VARIANT.as_dict(), "variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    del repo_root, spec, artifacts_dir
    return probe_baseline_contract(device_index=0)


def benchmark(repo_root, spec, artifacts_dir):
    del repo_root, spec
    report = probe_baseline_contract(device_index=0)
    if report["status"] == "ok":
        report = dict(report)
        report["timing_protocol"] = {"warmups": 20, "measure_iters": 100, "metric": "median_ms"}
        report["benchmark_note"] = (
            "Contract probe is green. Full repeated parity timing is deferred until the PTO port exists."
        )
        report["checked_at_epoch_s"] = time.time()
    report_path = Path(artifacts_dir) / "ops_transformer_grouped_matmul_swiglu_quant_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
