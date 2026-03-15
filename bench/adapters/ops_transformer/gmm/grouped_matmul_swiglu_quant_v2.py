from __future__ import annotations

import json
from pathlib import Path

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.gmm.grouped_matmul_swiglu_quant_v2.runtime import VARIANT, VARIANTS, baseline_blocker


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "gmm", "grouped_matmul_swiglu_quant_v2", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_grouped_matmul_swiglu_quant_v2"
    summary["seed_variant"] = {"default": VARIANT.as_dict(), "variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    del repo_root, spec, artifacts_dir
    return baseline_blocker(device_index=0)


def benchmark(repo_root, spec, artifacts_dir):
    del repo_root, spec
    report = baseline_blocker(device_index=0)
    report_path = Path(artifacts_dir) / "ops_transformer_grouped_matmul_swiglu_quant_v2_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
