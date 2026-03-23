#!/usr/bin/env python3
"""Classify active PTO kernels by parity state and scalar-hot-path usage."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from pto_kernels.config import repo_root


REPO_ROOT = repo_root()
REPORTS_DIR = REPO_ROOT / "bench" / "reports"
REGRESSION_JSON = REPORTS_DIR / "regression_latest.json"
INVENTORY_PATH = REPO_ROOT / "bench" / "kernel_inventory.yaml"
GAP_BOARD_PATH = REPO_ROOT / "bench" / "gap_board.yaml"
LATEST_JSON = REPORTS_DIR / "kernel_state_matrix_latest.json"
LATEST_MD = REPORTS_DIR / "kernel_state_matrix_latest.md"

SCALAR_PATTERNS = {
    "load_scalar": re.compile(r"\b(?:pto\.)?load_scalar\s*\("),
    "store_scalar": re.compile(r"\b(?:pto\.)?store_scalar\s*\("),
    "scalar_select": re.compile(r"\bpto\.select\s*\("),
}

SHARED_FILES = {
    ("attention", "flash_attention_score"): [
        REPO_ROOT / "python" / "pto_kernels" / "ops" / "attention" / "common.py",
    ],
    ("attention", "fused_infer_attention_score"): [
        REPO_ROOT / "python" / "pto_kernels" / "ops" / "attention" / "common.py",
    ],
    ("ffn", "ffn"): [
        REPO_ROOT / "python" / "pto_kernels" / "ops" / "ffn" / "common.py",
    ],
    ("ffn", "swin_attention_ffn"): [
        REPO_ROOT / "python" / "pto_kernels" / "ops" / "ffn" / "common.py",
    ],
}

PRIMARY_CLASS_ORDER = (
    "blocked by PTOAS lowering",
    "blocked by PTODSL surface",
    "blocked by pto-isa/backend capability",
    "blocked by PTO correctness gap",
    "blocked by host baseline/runtime gap",
    "green but scalar-heavy",
    "green + tile-first",
)


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_inventory() -> dict[str, dict[str, Any]]:
    inventory = _load_yaml(INVENTORY_PATH)
    included = inventory.get("included", [])
    return {item["name"]: item for item in included if isinstance(item, dict) and "name" in item}


def _load_gap_index() -> dict[str, list[dict[str, Any]]]:
    board = _load_yaml(GAP_BOARD_PATH)
    gaps = board.get("gaps", []) if isinstance(board, dict) else []

    by_kernel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for gap in gaps:
        if not isinstance(gap, dict):
            continue
        for kernel in (gap.get("blocking_kernels") or []):
            by_kernel[str(kernel)].append(gap)
    return by_kernel


def _scan_scalar_patterns(family: str, kernel: str) -> list[str]:
    kernel_dir = REPO_ROOT / "python" / "pto_kernels" / "ops" / family / kernel
    files: list[Path] = []
    if kernel_dir.is_dir():
        files.extend(sorted(kernel_dir.glob("*.py")))
    files.extend(SHARED_FILES.get((family, kernel), []))

    matches: list[str] = []
    seen = set()
    for path in files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT)
        for pattern_name, pattern in SCALAR_PATTERNS.items():
            count = len(pattern.findall(text))
            if count <= 0:
                continue
            key = (str(rel), pattern_name, count)
            if key in seen:
                continue
            seen.add(key)
            matches.append(f"{rel}:{pattern_name}x{count}")
    return matches


def _pick_primary_class(
    *,
    baseline_status: str,
    pto_status: str,
    pto_correctness_passes: bool | None,
    scalar_matches: list[str],
    kernel_gaps: list[dict[str, Any]],
) -> str:
    gap_components = {str(gap.get("component", "")).lower() for gap in kernel_gaps}
    if pto_status == "blocked":
        if "ptoas" in gap_components:
            return "blocked by PTOAS lowering"
        if "ptodsl" in gap_components:
            return "blocked by PTODSL surface"
        if "pto-isa" in gap_components:
            return "blocked by pto-isa/backend capability"
        if gap_components & {"runtime", "ops-transformer"}:
            return "blocked by host baseline/runtime gap"
        return "blocked by PTOAS lowering"
    if pto_status == "ok" and pto_correctness_passes is False:
        return "blocked by PTO correctness gap"
    if baseline_status == "blocked":
        return "blocked by host baseline/runtime gap"
    if scalar_matches:
        return "green but scalar-heavy"
    return "green + tile-first"


def _ratio_range(variants: list[dict[str, Any]]) -> str:
    ratios = [
        float(item["baseline_over_pto_pct"])
        for item in variants
        if item.get("baseline_over_pto_pct") is not None
    ]
    if not ratios:
        return "n/a"
    return f"{min(ratios):.1f}%..{max(ratios):.1f}%"


def _block_usage(report_path: Path) -> dict[str, Any] | None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report.get("pto", {}).get("benchmark", {}).get("block_utilization")


def _benchmark_correctness(report_path: Path, side: str) -> bool | None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    benchmark = report.get(side, {}).get("benchmark", {})
    correctness = benchmark.get("correctness")
    if isinstance(correctness, dict) and "passes" in correctness:
        return bool(correctness["passes"])
    return None


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Kernel State Matrix",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "| Kernel | Wave | Class | Scalar Hot Path | Block Use | Baseline | PTO | Baseline / PTO | Latest |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for item in payload["kernels"]:
        block_use = "n/a"
        if item["block_utilization"]:
            block_use = (
                f"{item['block_utilization'].get('requested_block_dim', 'n/a')} blocks, "
                f"all={item['block_utilization'].get('uses_all_blocks', False)}"
            )
        scalar_text = "yes" if item["scalar_hot_path"] else "no"
        lines.append(
            "| {name} | {wave} | {klass} | {scalar} | {blocks} | {baseline} | {pto} | {ratio} | [report]({report}) |".format(
                name=item["name"],
                wave=item["wave"],
                klass=item["primary_class"],
                scalar=scalar_text,
                blocks=block_use,
                baseline=item["baseline_status"],
                pto=item["pto_status"],
                ratio=item["baseline_over_pto_range"],
                report=Path(item["report_path"]).relative_to(REPO_ROOT),
            )
        )
        if item["scalar_matches"]:
            lines.append(f"|  |  | scalar matches: `{'; '.join(item['scalar_matches'])}` |  |  |  |  |  |  |")
        if item["gap_ids"]:
            lines.append(f"|  |  | gap ids: `{', '.join(item['gap_ids'])}` |  |  |  |  |  |  |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    regression = json.loads(REGRESSION_JSON.read_text(encoding="utf-8"))
    inventory = _load_inventory()
    gap_index = _load_gap_index()

    kernels = []
    for kernel in regression.get("kernels", []):
        name = kernel["name"]
        inventory_item = inventory.get(name, {})
        family = kernel["family"]
        gaps = gap_index.get(name, [])
        scalar_matches = _scan_scalar_patterns(family, name)
        report_path = Path(kernel["report_path"])
        kernels.append(
            {
                "name": name,
                "family": family,
                "wave": inventory_item.get("wave", "unknown"),
                "phase": inventory_item.get("phase", "unknown"),
                "inventory_status": inventory_item.get("status", "unknown"),
                "report_path": str(report_path),
                "baseline_status": kernel["baseline_status"],
                "pto_status": kernel["pto_status"],
                "baseline_reason": kernel.get("baseline_reason"),
                "pto_reason": kernel.get("pto_reason"),
                "scalar_hot_path": bool(scalar_matches),
                "scalar_matches": scalar_matches,
                "gap_ids": [str(gap.get("id")) for gap in gaps],
                "gap_components": [str(gap.get("component")) for gap in gaps],
                "baseline_over_pto_range": _ratio_range(kernel.get("variants", [])),
                "block_utilization": _block_usage(report_path),
                "pto_correctness_passes": _benchmark_correctness(report_path, "pto"),
                "primary_class": _pick_primary_class(
                    baseline_status=kernel["baseline_status"],
                    pto_status=kernel["pto_status"],
                    pto_correctness_passes=_benchmark_correctness(report_path, "pto"),
                    scalar_matches=scalar_matches,
                    kernel_gaps=gaps,
                ),
            }
        )

    kernels.sort(
        key=lambda item: (
            PRIMARY_CLASS_ORDER.index(item["primary_class"])
            if item["primary_class"] in PRIMARY_CLASS_ORDER
            else len(PRIMARY_CLASS_ORDER),
            item["wave"],
            item["family"],
            item["name"],
        )
    )
    payload = {
        "generated_at": datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ"),
        "kernels": kernels,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamped_json = REPORTS_DIR / f"kernel_state_matrix_{payload['generated_at']}.json"
    timestamped_md = REPORTS_DIR / f"kernel_state_matrix_{payload['generated_at']}.md"
    timestamped_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(timestamped_md, payload)
    LATEST_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(LATEST_MD, payload)
    print(
        json.dumps(
            {
                "json": str(timestamped_json),
                "markdown": str(timestamped_md),
                "latest_json": str(LATEST_JSON),
                "latest_markdown": str(LATEST_MD),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
