from __future__ import annotations

import json
from pathlib import Path

from pto_kernels.bench.adapter_utils import describe_pto
from pto_kernels.ops.ffn.swin_transformer_ln_qkv_quant.runtime import pto_blocker


KERNEL = "python/pto_kernels/ops/ffn/swin_transformer_ln_qkv_quant/kernel.py"
META = "python/pto_kernels/ops/ffn/swin_transformer_ln_qkv_quant/meta.py"


def describe(repo_root, spec):
    return describe_pto(repo_root, KERNEL, META)


def compile_kernel(repo_root, spec, artifacts_dir):
    del repo_root, spec, artifacts_dir
    return pto_blocker()


def benchmark(repo_root, spec, artifacts_dir):
    del repo_root, spec
    report = pto_blocker()
    report_path = Path(artifacts_dir) / "ptodsl_swin_transformer_ln_qkv_quant_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
