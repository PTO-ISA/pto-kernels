"""Shared metadata helpers for planned PTO-DSL kernels."""

from __future__ import annotations

from typing import Any


DEFAULT_CHECKLIST = [
    "baseline_adapter",
    "pto_dsl_source",
    "artifact_trace",
    "correctness",
    "benchmark",
    "parity",
]


def planned_meta(
    *,
    family: str,
    name: str,
    wave: str,
    archetype: str,
    ops_transformer_path: str,
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "family": family,
        "name": name,
        "wave": wave,
        "archetype": archetype,
        "ops_transformer_path": ops_transformer_path,
        "status": "planned",
        "checklist": list(DEFAULT_CHECKLIST),
        "blockers": blockers,
    }
