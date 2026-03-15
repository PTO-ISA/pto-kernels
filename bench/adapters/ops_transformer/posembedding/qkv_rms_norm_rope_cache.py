from __future__ import annotations

import json
from pathlib import Path

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.posembedding.qkv_rms_norm_rope_cache.runtime import VARIANTS, baseline_available


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "posembedding", "qkv_rms_norm_rope_cache", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_qkv_rms_norm_rope_cache"
    summary["seed_variant"] = {"variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_baseline(repo_root, spec):
    if not baseline_available():
        return {
            "status": "blocked",
            "reason": "torch_npu does not expose npu_qkv_rms_norm_rope_cache on this environment.",
        }
    return {
        "status": "runtime_builtin",
        "entrypoint": "torch_npu.npu_qkv_rms_norm_rope_cache",
    }


def benchmark(repo_root, spec, artifacts_dir):
    report = {
        "status": "blocked",
        "entrypoint": "torch_npu.npu_qkv_rms_norm_rope_cache",
        "reason": "Baseline runtime entrypoint is unavailable on this host.",
        "variants": [variant.as_dict() for variant in VARIANTS],
    }
    report_path = Path(artifacts_dir) / "ops_transformer_qkv_rms_norm_rope_cache_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
