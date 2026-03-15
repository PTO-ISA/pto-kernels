from __future__ import annotations

import json
from pathlib import Path

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.ffn.swin_transformer_ln_qkv_quant.runtime import (
    VARIANTS,
    baseline_blocker,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "ffn", "swin_transformer_ln_qkv_quant", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_swin_transformer_ln_qkv_quant"
    summary["seed_variant"] = {"variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    del repo_root, spec, artifacts_dir
    return baseline_blocker()


def benchmark(repo_root, spec, artifacts_dir):
    del repo_root, spec
    report = baseline_blocker()
    report["baseline_limitations"] = [
        "No Python-visible torch_npu or torch.ops.npu entrypoint on this host",
        "Public ACLNN documentation marks the operator unsupported on Atlas A2 / 910B",
    ]
    report_path = Path(artifacts_dir) / "ops_transformer_swin_transformer_ln_qkv_quant_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
