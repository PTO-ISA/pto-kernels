from __future__ import annotations

import json
from pathlib import Path

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.ffn.swin_attention_ffn.runtime import VARIANTS, baseline_available


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "ffn", "swin_attention_ffn", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_swin_attention_ffn"
    summary["seed_variant"] = {"variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    del repo_root, spec, artifacts_dir
    if not baseline_available():
        return {
            "status": "blocked",
            "reason": "torch_npu does not expose npu_swin_attention_ffn on this host.",
        }
    return {"status": "runtime_builtin", "entrypoint": "torch_npu.npu_swin_attention_ffn"}


def benchmark(repo_root, spec, artifacts_dir):
    del repo_root, spec
    report = {
        "status": "blocked",
        "entrypoint": "torch_npu.npu_swin_attention_ffn",
        "reason": "Baseline runtime entrypoint is unavailable on this host.",
        "variants": [variant.as_dict() for variant in VARIANTS],
    }
    report_path = Path(artifacts_dir) / "ops_transformer_swin_attention_ffn_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
