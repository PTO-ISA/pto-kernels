#!/usr/bin/env python3
"""Probe host-side baseline contracts for quantized Wave 1 GMM kernels."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch
import torch_npu  # noqa: F401

from pto_kernels.config import repo_root
from pto_kernels.ops.gmm.grouped_matmul_swiglu_quant.runtime import probe_baseline_contract as probe_swiglu_quant_baseline
from pto_kernels.utils.env import detect_env


def _blocked(reason: str, **extra) -> dict[str, object]:
    payload: dict[str, object] = {"status": "blocked", "reason": reason}
    payload.update(extra)
    return payload


def _ok(**extra) -> dict[str, object]:
    payload: dict[str, object] = {"status": "ok"}
    payload.update(extra)
    return payload


def _has_op(name: str) -> bool:
    return hasattr(torch.ops.npu, name)


def probe_grouped_matmul_finalize_routing() -> dict[str, object]:
    if not _has_op("npu_grouped_matmul_finalize_routing"):
        return _blocked("torch_npu does not expose npu_grouped_matmul_finalize_routing")
    return _blocked(
        "Host baseline remains unstable: routed quantized tensors are required; ND int8 weights are rejected; "
        "supported W4A8-sized ND probes can segfault; NZ-format probing depends on working TBE Python deps.",
        entrypoint="torch_npu.npu_grouped_matmul_finalize_routing",
        probe_shapes=[
            {"x": [64, 128], "weight": [1, 128, 128], "weight_dtype": "int8_nd"},
            {"x": [64, 192], "weight": [1, 192, 128], "weight_dtype": "int32_w4a8_nd"},
        ],
    )


def probe_grouped_matmul_swiglu_quant() -> dict[str, object]:
    if not _has_op("npu_grouped_matmul_swiglu_quant"):
        return _blocked("torch_npu does not expose npu_grouped_matmul_swiglu_quant")
    return probe_swiglu_quant_baseline(device_index=0)


def probe_grouped_matmul_swiglu_quant_v2() -> dict[str, object]:
    if not _has_op("npu_grouped_matmul_swiglu_quant_v2"):
        return _blocked("torch_npu does not expose npu_grouped_matmul_swiglu_quant_v2")
    return _blocked(
        "Host baseline requires list-valued low-precision weight tensors plus FP8/scale contracts from the upstream "
        "ACLNN example. A stable minimal Python baseline slice is not reproduced yet.",
        entrypoint="torch_npu.npu_grouped_matmul_swiglu_quant_v2",
        probe_shapes=[
            {
                "x": [2048, 7168],
                "weight_list_item": [8, 7168, 4096],
                "weight_scale_list_item": [8, 112, 4096, 2],
                "x_scale": [2048, 112, 2],
                "group_list": [8],
                "output": [2048, 2048],
                "output_scale": [2048, 32, 2],
            }
        ],
    )


def probe_quant_grouped_matmul_inplace_add() -> dict[str, object]:
    if _has_op("npu_quant_grouped_matmul_inplace_add"):
        return _ok(entrypoint="torch_npu.npu_quant_grouped_matmul_inplace_add")
    return _blocked(
        "No torch_npu Python entrypoint is exposed on this host. The upstream kernel currently exists only through "
        "ACLNN/C++ host interfaces in ops-transformer tests and examples.",
        entrypoint="aclnnQuantGroupedMatmulInplaceAdd",
        probe_shapes=[
            {
                "x1": [512, 96],
                "x2": [512, 128],
                "scale1": [4],
                "scale2": [4, 128],
                "y": [4, 96, 128],
                "group_list": [4],
            }
        ],
    )


def main() -> int:
    results = {
        "environment": json.loads(detect_env().to_json()),
        "kernels": {
            "grouped_matmul_finalize_routing": probe_grouped_matmul_finalize_routing(),
            "grouped_matmul_swiglu_quant": probe_grouped_matmul_swiglu_quant(),
            "grouped_matmul_swiglu_quant_v2": probe_grouped_matmul_swiglu_quant_v2(),
            "quant_grouped_matmul_inplace_add": probe_quant_grouped_matmul_inplace_add(),
        },
    }

    root = repo_root() / "bench" / "reports"
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "quantized_gmm_contracts_latest.json"
    md_path = root / "quantized_gmm_contracts_latest.md"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    lines = ["# Quantized GMM Contract Probe", ""]
    for name, report in results["kernels"].items():
        lines.append(f"## {name}")
        lines.append(f"- status: `{report['status']}`")
        lines.append(f"- reason: {report['reason']}")
        if "entrypoint" in report:
            lines.append(f"- entrypoint: `{report['entrypoint']}`")
        if "probe_shapes" in report:
            lines.append(f"- probe_shapes: `{json.dumps(report['probe_shapes'], sort_keys=True)}`")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"json": str(json_path), "md": str(md_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
