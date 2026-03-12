#!/usr/bin/env python3
"""Run repeated parity measurements for the current default seed kernels."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pto_kernels.bench.runner import BenchmarkRunner
from pto_kernels.config import repo_root


REPO_ROOT = repo_root()
REPORTS_DIR = REPO_ROOT / "bench" / "reports"
LATEST_JSON = REPORTS_DIR / "kernel_parity_latest.json"
LATEST_MD = REPORTS_DIR / "kernel_parity_latest.md"
SPECS = (
    ("grouped_matmul", "bench/specs/gmm/grouped_matmul.yaml"),
    ("apply_rotary_pos_emb", "bench/specs/posembedding/apply_rotary_pos_emb.yaml"),
    ("ffn", "bench/specs/ffn/ffn.yaml"),
    ("moe_token_permute", "bench/specs/moe/moe_token_permute.yaml"),
    ("flash_attention_score", "bench/specs/attention/flash_attention_score.yaml"),
    ("matmul_reduce_scatter", "bench/specs/mc2/matmul_reduce_scatter.yaml"),
)


def metric(report: dict, side: str, key: str):
    benchmark = report.get(side, {}).get("benchmark", {})
    if key == "median":
        return benchmark.get("timings_ms", {}).get("median")
    if key == "correct":
        return bool(benchmark.get("correctness", {}).get("passes"))
    return benchmark.get(key)


def efficiency_pct(baseline: float | None, pto: float | None) -> float | None:
    if baseline is None or pto is None or baseline <= 0:
        return None
    return baseline / pto * 100.0


def fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def _variant_key(variant: dict[str, Any]) -> str:
    return json.dumps(variant, sort_keys=True)


def _extract_variant_reports(report: dict[str, Any], side: str) -> list[dict[str, Any]]:
    benchmark = report.get(side, {}).get("benchmark", {})
    variant_reports = benchmark.get("variant_reports")
    if isinstance(variant_reports, list) and variant_reports:
        return variant_reports

    variant = benchmark.get("variant")
    if isinstance(variant, dict):
        return [
            {
                "variant": variant,
                "shape_summary": benchmark.get("shape_summary"),
                "timings_ms": benchmark.get("timings_ms", {}),
                "correctness": benchmark.get("correctness", {}),
            }
        ]
    return []


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Kernel Parity Summary",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Rounds per kernel: `{summary['rounds']}`",
        "",
        "| Kernel | Input Shape | Baseline ms | PTO ms | Baseline / PTO | Correctness | Latest |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for kernel in summary["kernels"]:
        for variant in kernel["variants"]:
            correctness = "pass" if variant["all_correct"] else "fail"
            lines.append(
                "| {name} | `{shape}` | {baseline} | {pto} | {pct} | {correctness} | [report]({latest}) |".format(
                    name=kernel["name"],
                    shape=json.dumps(variant["shape_summary"], sort_keys=True),
                    baseline=fmt_ms(variant["baseline_median_ms"]),
                    pto=fmt_ms(variant["pto_median_ms"]),
                    pct=fmt_pct(variant["baseline_over_pto_pct"]),
                    correctness=correctness,
                    latest=Path(kernel["latest_report_path"]).relative_to(REPO_ROOT),
                )
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    runner = BenchmarkRunner()
    generated_at = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    kernels = []

    for name, spec in SPECS:
        rounds = []
        for round_index in range(1, args.rounds + 1):
            report = runner.run(spec)
            baseline_ms = metric(report, "baseline", "median")
            pto_ms = metric(report, "pto", "median")
            baseline_variants = _extract_variant_reports(report, "baseline")
            pto_variants = _extract_variant_reports(report, "pto")
            variant_rows = []
            pto_by_key = {
                _variant_key(item["variant"]): item
                for item in pto_variants
                if isinstance(item.get("variant"), dict)
            }
            for baseline_variant in baseline_variants:
                variant = baseline_variant["variant"]
                key = _variant_key(variant)
                pto_variant = pto_by_key.get(key)
                variant_rows.append(
                    {
                        "variant": variant,
                        "shape_summary": baseline_variant.get("shape_summary"),
                        "baseline_median_ms": baseline_variant.get("timings_ms", {}).get("median"),
                        "pto_median_ms": None if pto_variant is None else pto_variant.get("timings_ms", {}).get("median"),
                        "baseline_over_pto_pct": None
                        if pto_variant is None
                        else efficiency_pct(
                            baseline_variant.get("timings_ms", {}).get("median"),
                            pto_variant.get("timings_ms", {}).get("median"),
                        ),
                    }
                )
            rounds.append(
                {
                    "round": round_index,
                    "report_path": report["report_path"],
                    "latest_report_path": report["latest_report_path"],
                    "baseline_median_ms": baseline_ms,
                    "pto_median_ms": pto_ms,
                    "baseline_over_pto_pct": efficiency_pct(baseline_ms, pto_ms),
                    "baseline_correct": metric(report, "baseline", "correct"),
                    "pto_correct": metric(report, "pto", "correct"),
                    "baseline_status": report["baseline"]["benchmark"]["status"],
                    "pto_status": report["pto"]["benchmark"]["status"],
                    "variant_rows": variant_rows,
                }
            )
        baseline_values = [round_info["baseline_median_ms"] for round_info in rounds if round_info["baseline_median_ms"] is not None]
        pto_values = [round_info["pto_median_ms"] for round_info in rounds if round_info["pto_median_ms"] is not None]
        baseline_median_ms = statistics.median(baseline_values) if baseline_values else None
        pto_median_ms = statistics.median(pto_values) if pto_values else None
        variant_map: dict[str, dict[str, Any]] = {}
        for round_info in rounds:
            for variant_row in round_info["variant_rows"]:
                key = _variant_key(variant_row["variant"])
                entry = variant_map.setdefault(
                    key,
                    {
                        "variant": variant_row["variant"],
                        "shape_summary": variant_row["shape_summary"],
                        "baseline_values": [],
                        "pto_values": [],
                    },
                )
                if variant_row["baseline_median_ms"] is not None:
                    entry["baseline_values"].append(variant_row["baseline_median_ms"])
                if variant_row["pto_median_ms"] is not None:
                    entry["pto_values"].append(variant_row["pto_median_ms"])

        variants = []
        for entry in variant_map.values():
            baseline_variant_median = (
                statistics.median(entry["baseline_values"]) if entry["baseline_values"] else None
            )
            pto_variant_median = statistics.median(entry["pto_values"]) if entry["pto_values"] else None
            variants.append(
                {
                    "variant": entry["variant"],
                    "shape_summary": entry["shape_summary"],
                    "baseline_median_ms": baseline_variant_median,
                    "pto_median_ms": pto_variant_median,
                    "baseline_over_pto_pct": efficiency_pct(
                        baseline_variant_median,
                        pto_variant_median,
                    ),
                    "all_correct": all(
                        round_info["baseline_correct"] and round_info["pto_correct"] for round_info in rounds
                    ),
                }
            )
        variants.sort(key=lambda item: _variant_key(item["variant"]))
        kernels.append(
            {
                "name": name,
                "spec": spec,
                "rounds": rounds,
                "latest_report_path": rounds[-1]["latest_report_path"],
                "variants": variants,
                "aggregate": {
                    "baseline_median_ms": baseline_median_ms,
                    "pto_median_ms": pto_median_ms,
                    "baseline_over_pto_pct": efficiency_pct(baseline_median_ms, pto_median_ms),
                    "all_correct": all(round_info["baseline_correct"] and round_info["pto_correct"] for round_info in rounds),
                },
            }
        )

    summary = {"generated_at": generated_at, "rounds": args.rounds, "kernels": kernels}
    json_path = REPORTS_DIR / f"kernel_parity_{generated_at}.json"
    md_path = REPORTS_DIR / f"kernel_parity_{generated_at}.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, summary)
    LATEST_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(LATEST_MD, summary)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
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
