from __future__ import annotations

import json
from pathlib import Path

from pto_kernels.bench.adapter_utils import describe_pto
from pto_kernels.ops.attention.recurrent_gated_delta_rule.runtime import VARIANTS


KERNEL = "python/pto_kernels/ops/attention/recurrent_gated_delta_rule/kernel.py"
META = "python/pto_kernels/ops/attention/recurrent_gated_delta_rule/meta.py"


def describe(repo_root, spec):
    return describe_pto(repo_root, KERNEL, META)


def compile_kernel(repo_root, spec, artifacts_dir):
    report = {
        "status": "blocked",
        "kernel_path": str(repo_root / KERNEL),
        "output_dir": str(artifacts_dir),
        "reason": "Tile-first PTODSL/PTOAS recurrent state update surface is not available for recurrent_gated_delta_rule.",
        "blocker_id": "ptodsl-recurrent-state-update-primitives",
    }
    return report


def benchmark(repo_root, spec, artifacts_dir):
    report = {
        "status": "blocked",
        "variants": [variant.as_dict() for variant in VARIANTS],
        "reason": (
            "PTO path is intentionally blocked: the current PTODSL/PTOAS stack does not expose a tile-first recurrent "
            "state update surface for BF16 state matvec + outer-product update + ragged token/state mapping, and the "
            "scalar-loop fallback is intentionally not used in the migration program."
        ),
        "blocker_id": "ptodsl-recurrent-state-update-primitives",
        "reference_contract": "nd_recurrent_no_gk",
    }
    report_path = Path(artifacts_dir) / "ptodsl_recurrent_gated_delta_rule_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
