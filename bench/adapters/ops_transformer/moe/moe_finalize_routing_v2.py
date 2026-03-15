from __future__ import annotations

import json
from pathlib import Path

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.moe.moe_finalize_routing_v2.runtime import VARIANT, VARIANTS, baseline_available


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "moe", "moe_finalize_routing_v2", spec.inventory_ref)
    summary["runtime_entrypoint"] = (
        "torch_npu.npu_moe_finalize_routing_v2 | torch.ops.npu.npu_moe_finalize_routing_v2"
    )
    summary["seed_variant"] = {"default": VARIANT.as_dict(), "variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    del repo_root, spec, artifacts_dir
    if not baseline_available():
        return {
            "status": "blocked",
            "reason": (
                "This host does not expose torch_npu.npu_moe_finalize_routing_v2 or "
                "torch.ops.npu.npu_moe_finalize_routing_v2."
            ),
        }
    return {
        "status": "runtime_builtin",
        "entrypoint": "torch_npu.npu_moe_finalize_routing_v2 | torch.ops.npu.npu_moe_finalize_routing_v2",
        "note": (
            "The constrained slice fixes topK=1, x1Optional/x2Optional/biasOptional/scalesOptional present, "
            "dropPadMode=0, and 2D expandedX."
        ),
    }


def benchmark(repo_root, spec, artifacts_dir):
    del repo_root, spec
    report = {
        "status": "blocked",
        "entrypoint": "torch_npu.npu_moe_finalize_routing_v2 | torch.ops.npu.npu_moe_finalize_routing_v2",
        "reason": "Baseline runtime entrypoint is unavailable on this host.",
        "variants": [variant.as_dict() for variant in VARIANTS],
    }
    report_path = Path(artifacts_dir) / "ops_transformer_moe_finalize_routing_v2_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
