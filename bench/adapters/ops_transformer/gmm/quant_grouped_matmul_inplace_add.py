from __future__ import annotations

import json
from pathlib import Path

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.gmm.quant_grouped_matmul_inplace_add.runtime import (
    VARIANT,
    VARIANTS,
    baseline_blocker,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "gmm", "quant_grouped_matmul_inplace_add", spec.inventory_ref)
    summary["runtime_entrypoint"] = "aclnnQuantGroupedMatmulInplaceAdd"
    summary["seed_variant"] = {"default": VARIANT.as_dict(), "variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    del repo_root, spec, artifacts_dir
    return baseline_blocker(device_index=0)


def benchmark(repo_root, spec, artifacts_dir):
    del repo_root, spec
    report = baseline_blocker(device_index=0)
    report_path = Path(artifacts_dir) / "ops_transformer_quant_grouped_matmul_inplace_add_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
