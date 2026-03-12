#!/usr/bin/env python3
"""Summarize the latest phase-1 seed reports into a single audit artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "bench" / "results"
OUTPUT_JSON = REPO_ROOT / "bench" / "reports" / "phase1_seed_audit.json"
OUTPUT_MD = REPO_ROOT / "bench" / "reports" / "phase1_seed_audit.md"


@dataclass(frozen=True)
class SeedRef:
    family: str
    name: str

    @property
    def results_dir(self) -> Path:
        return RESULTS_ROOT / self.family / self.name

    @property
    def label(self) -> str:
        return f"{self.family}/{self.name}"


SEEDS = (
    SeedRef("posembedding", "apply_rotary_pos_emb"),
    SeedRef("gmm", "grouped_matmul"),
    SeedRef("ffn", "ffn"),
    SeedRef("moe", "moe_token_permute"),
    SeedRef("attention", "flash_attention_score"),
    SeedRef("mc2", "matmul_reduce_scatter"),
)


def _latest_report(seed: SeedRef) -> Path:
    candidates = [path / "report.json" for path in seed.results_dir.iterdir() if path.is_dir()]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(f"No report.json files found for {seed.label} in {seed.results_dir}")
    return max(existing, key=lambda path: path.stat().st_mtime)


def _extract(seed: SeedRef) -> dict[str, object]:
    report_path = _latest_report(seed)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    baseline = report.get("baseline", {}).get("benchmark", {})
    pto = report.get("pto", {}).get("benchmark", {})
    baseline_correctness = baseline.get("correctness", {})
    pto_correctness = pto.get("correctness", {})

    return {
        "seed": seed.label,
        "report_path": str(report_path),
        "baseline": {
            "status": baseline.get("status"),
            "median_ms": baseline.get("timings_ms", {}).get("median"),
            "passes": baseline_correctness.get("passes"),
            "max_abs_diff": baseline_correctness.get("max_abs_diff"),
        },
        "pto": {
            "status": pto.get("status"),
            "median_ms": pto.get("timings_ms", {}).get("median"),
            "passes": pto_correctness.get("passes"),
            "max_abs_diff": pto_correctness.get("max_abs_diff"),
            "entrypoint": pto.get("entrypoint"),
        },
    }


def _render_markdown(entries: list[dict[str, object]]) -> str:
    lines = [
        "# Phase 1 Seed Audit",
        "",
        "| Seed | Baseline | PTO | Baseline ms | PTO ms | PTO max abs diff |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for entry in entries:
        lines.append(
            "| {seed} | {b_status}/{b_pass} | {p_status}/{p_pass} | {b_ms:.4f} | {p_ms:.4f} | {p_diff} |".format(
                seed=entry["seed"],
                b_status=entry["baseline"]["status"],
                b_pass="pass" if entry["baseline"]["passes"] else "fail",
                p_status=entry["pto"]["status"],
                p_pass="pass" if entry["pto"]["passes"] else "fail",
                b_ms=float(entry["baseline"]["median_ms"]),
                p_ms=float(entry["pto"]["median_ms"]),
                p_diff=entry["pto"]["max_abs_diff"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    entries = [_extract(seed) for seed in SEEDS]
    audit = {
        "phase1_complete": all(
            entry["baseline"]["status"] == "ok"
            and entry["baseline"]["passes"] is True
            and entry["pto"]["status"] == "ok"
            and entry["pto"]["passes"] is True
            for entry in entries
        ),
        "seeds": entries,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    OUTPUT_MD.write_text(_render_markdown(entries), encoding="utf-8")
    print(json.dumps({"json": str(OUTPUT_JSON), "markdown": str(OUTPUT_MD), "phase1_complete": audit["phase1_complete"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
