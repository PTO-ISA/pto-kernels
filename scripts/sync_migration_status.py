#!/usr/bin/env python3
"""Sync inventory and migration checklist from the latest kernel state matrix."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from pto_kernels.config import repo_root


REPO_ROOT = repo_root()
STATE_MATRIX_PATH = REPO_ROOT / "bench" / "reports" / "kernel_state_matrix_latest.json"
INVENTORY_PATH = REPO_ROOT / "bench" / "kernel_inventory.yaml"
CHECKLIST_PATH = REPO_ROOT / "checklists" / "910b_ai_core_migration.md"

CHECKLIST_LEGEND = [
    "",
    "Legend:",
    "- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO",
    "- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy",
    "- `[ ]` not started",
    "",
]


def _load_state_matrix() -> list[dict]:
    payload = json.loads(STATE_MATRIX_PATH.read_text(encoding="utf-8"))
    kernels = payload.get("kernels", [])
    return [item for item in kernels if isinstance(item, dict) and item.get("name")]


def _inventory_status(item: dict) -> str:
    klass = item.get("primary_class", "")
    pto_ok = item.get("pto_status") == "ok"
    scalar_hot = bool(item.get("scalar_hot_path"))

    if klass == "green + tile-first":
        return "green"
    if klass == "green but scalar-heavy":
        return "scalar-heavy"
    if klass == "blocked by host baseline/runtime gap" and pto_ok:
        return "scalar-heavy" if scalar_hot else "external-blocked"
    if klass.startswith("blocked by "):
        return "blocked"
    return "prototype"


def _checklist_marker(item: dict) -> str:
    klass = item.get("primary_class", "")
    pto_ok = item.get("pto_status") == "ok"
    scalar_hot = bool(item.get("scalar_hot_path"))

    if klass == "green + tile-first":
        return "x"
    if klass == "blocked by host baseline/runtime gap" and pto_ok and not scalar_hot:
        return "x"
    return "~"


def _sync_inventory(kernels: list[dict]) -> dict[str, str]:
    status_by_name = {item["name"]: _inventory_status(item) for item in kernels}
    lines = INVENTORY_PATH.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    entry_re = re.compile(r"^(?P<prefix>\s*-\s*\{name:\s*)(?P<name>[^,]+)(?P<middle>.*?,\sstatus:\s*)(?P<status>[^,}]+)(?P<suffix>.*\}\s*)$")

    for line in lines:
        match = entry_re.match(line)
        if not match:
            out.append(line)
            continue
        name = match.group("name").strip()
        if name not in status_by_name:
            out.append(line)
            continue
        out.append(
            f"{match.group('prefix')}{name}{match.group('middle')}{status_by_name[name]}{match.group('suffix')}"
        )

    INVENTORY_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    return status_by_name


def _sync_checklist(kernels: list[dict]) -> dict[str, str]:
    marker_by_kernel = {f"{item['family']}/{item['name']}": _checklist_marker(item) for item in kernels}
    lines = CHECKLIST_PATH.read_text(encoding="utf-8").splitlines()
    out: list[str] = []

    legend_inserted = False
    item_re = re.compile(r"^(?P<prefix>-\s+\[)(?P<mark>.)(?P<mid>\]\s+`)(?P<kernel>[^`]+)(?P<suffix>`.*)$")

    for idx, line in enumerate(lines):
        out.append(line)
        if not legend_inserted and line.startswith("# 910B AI Core Migration Checklist"):
            if idx + 1 >= len(lines) or not lines[idx + 1].startswith("Legend:"):
                out.extend(CHECKLIST_LEGEND)
            legend_inserted = True

    final: list[str] = []
    for line in out:
        match = item_re.match(line)
        if not match:
            final.append(line)
            continue
        kernel = match.group("kernel")
        if kernel not in marker_by_kernel:
            final.append(line)
            continue
        final.append(
            f"{match.group('prefix')}{marker_by_kernel[kernel]}{match.group('mid')}{kernel}{match.group('suffix')}"
        )

    CHECKLIST_PATH.write_text("\n".join(final) + "\n", encoding="utf-8")
    return marker_by_kernel


def _validate_gap_board() -> None:
    gap_board = REPO_ROOT / "bench" / "gap_board.yaml"
    yaml.safe_load(gap_board.read_text(encoding="utf-8"))


def main() -> int:
    _validate_gap_board()
    kernels = _load_state_matrix()
    statuses = _sync_inventory(kernels)
    markers = _sync_checklist(kernels)
    print(
        json.dumps(
            {
                "inventory_path": str(INVENTORY_PATH),
                "checklist_path": str(CHECKLIST_PATH),
                "kernels_synced": len(kernels),
                "status_counts": {
                    status: list(statuses.values()).count(status)
                    for status in sorted(set(statuses.values()))
                },
                "checklist_counts": {
                    marker: list(markers.values()).count(marker)
                    for marker in sorted(set(markers.values()))
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
