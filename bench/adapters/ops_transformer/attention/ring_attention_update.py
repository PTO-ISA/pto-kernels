from __future__ import annotations

import json
from pathlib import Path

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.attention.ring_attention_update.runtime import VARIANT, VARIANTS, baseline_available


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "attention", "ring_attention_update", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch.ops.npu.npu_ring_attention_update"
    summary["seed_variant"] = {"default": VARIANT.as_dict(), "variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    del repo_root, spec, artifacts_dir
    if not baseline_available():
        return {
            "status": "blocked",
            "reason": "This host does not expose torch.ops.npu.npu_ring_attention_update.",
        }
    return {
        "status": "runtime_builtin",
        "entrypoint": "torch.ops.npu.npu_ring_attention_update",
        "note": "The constrained slice fixes TND layout, N=1, and repeated last-dim-8 softmax max/sum tensors.",
    }


def benchmark(repo_root, spec, artifacts_dir):
    del repo_root, spec
    report = {
        "status": "blocked",
        "entrypoint": "torch.ops.npu.npu_ring_attention_update",
        "reason": "Baseline runtime entrypoint is unavailable on this host.",
        "variants": [variant.as_dict() for variant in VARIANTS],
    }
    report_path = Path(artifacts_dir) / "ops_transformer_ring_attention_update_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
