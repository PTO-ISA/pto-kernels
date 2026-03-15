from __future__ import annotations

import json
from pathlib import Path

from pto_kernels.bench.adapter_utils import compile_pto_kernel, describe_pto
from pto_kernels.ops.gmm.grouped_matmul_finalize_routing.runtime import VARIANTS


KERNEL = "python/pto_kernels/ops/gmm/grouped_matmul_finalize_routing/kernel.py"
META = "python/pto_kernels/ops/gmm/grouped_matmul_finalize_routing/meta.py"


def describe(repo_root, spec):
    return describe_pto(repo_root, KERNEL, META)


def compile_kernel(repo_root, spec, artifacts_dir):
    return compile_pto_kernel(repo_root, KERNEL, artifacts_dir)


def benchmark(repo_root, spec, artifacts_dir):
    del repo_root, spec
    report = {
        "status": "blocked",
        "reason": (
            "PTO port is intentionally deferred until the baseline routed quantized contract and weight "
            "storage layout are reproducible on this host."
        ),
        "variants": [variant.as_dict() for variant in VARIANTS],
    }
    report_path = Path(artifacts_dir) / "ptodsl_grouped_matmul_finalize_routing_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
