#!/usr/bin/env python3
"""Run repeated regression parity measurements for all active PTO kernels."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from pto_kernels.bench.runner import BenchmarkRunner
from pto_kernels.config import repo_root


REPO_ROOT = repo_root()
REPORTS_DIR = REPO_ROOT / "bench" / "reports"
REGRESSION_LIST = REPO_ROOT / "bench" / "regression_kernels.yaml"
LATEST_JSON = REPORTS_DIR / "regression_parity_latest.json"
LATEST_MD = REPORTS_DIR / "regression_parity_latest.md"


def _load_specs() -> list[str]:
    specs = yaml.safe_load(REGRESSION_LIST.read_text(encoding="utf-8"))
    if not isinstance(specs, list) or not all(isinstance(item, str) for item in specs):
        raise ValueError(f"Expected a list of spec paths in {REGRESSION_LIST}")
    return specs


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
                "status": benchmark.get("status"),
            }
        ]
    return []


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def _ratio_pct(baseline: float | None, pto: float | None) -> float | None:
    if baseline is None or pto is None or pto <= 0:
        return None
    return baseline / pto * 100.0


def _shape_text(item: dict[str, Any] | None) -> str:
    if not item:
        return "n/a"
    return json.dumps(item, sort_keys=True)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Regression Parity Summary",
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
                "| {name} | `{shape}` | {baseline} | {pto} | {ratio} | {correctness} | [report]({report}) |".format(
                    name=kernel["name"],
                    shape=_shape_text(variant["shape_summary"]),
                    baseline=_fmt_ms(variant["baseline_median_ms"]),
                    pto=_fmt_ms(variant["pto_median_ms"]),
                    ratio=_fmt_pct(variant["baseline_over_pto_pct"]),
                    correctness=correctness,
                    report=Path(kernel["latest_report_path"]).relative_to(REPO_ROOT),
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
    kernels: list[dict[str, Any]] = []

    for spec_path in _load_specs():
        rounds: list[dict[str, Any]] = []
        for round_index in range(1, args.rounds + 1):
            report = runner.run(spec_path)
            baseline_bench = report.get("baseline", {}).get("benchmark", {})
            pto_bench = report.get("pto", {}).get("benchmark", {})
            baseline_variants = _extract_variant_reports(report, "baseline")
            pto_variants = _extract_variant_reports(report, "pto")
            pto_by_key = {
                _variant_key(item["variant"]): item
                for item in pto_variants
                if isinstance(item.get("variant"), dict)
            }
            variant_rows = []
            for baseline_variant in baseline_variants:
                key = _variant_key(baseline_variant["variant"])
                pto_variant = pto_by_key.get(key)
                baseline_ms = baseline_variant.get("timings_ms", {}).get("median")
                pto_ms = None if pto_variant is None else pto_variant.get("timings_ms", {}).get("median")
                variant_rows.append(
                    {
                        "variant": baseline_variant["variant"],
                        "shape_summary": baseline_variant.get("shape_summary"),
                        "baseline_median_ms": baseline_ms,
                        "pto_median_ms": pto_ms,
                        "baseline_over_pto_pct": _ratio_pct(baseline_ms, pto_ms),
                        "baseline_correct": bool(baseline_variant.get("correctness", {}).get("passes")),
                        "pto_correct": bool(pto_variant and pto_variant.get("correctness", {}).get("passes")),
                    }
                )

            rounds.append(
                {
                    "round": round_index,
                    "report_path": report["report_path"],
                    "latest_report_path": report["latest_report_path"],
                    "baseline_status": baseline_bench.get("status"),
                    "pto_status": pto_bench.get("status"),
                    "variant_rows": variant_rows,
                }
            )

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
                        "baseline_ok": [],
                        "pto_ok": [],
                    },
                )
                if variant_row["baseline_median_ms"] is not None:
                    entry["baseline_values"].append(variant_row["baseline_median_ms"])
                if variant_row["pto_median_ms"] is not None:
                    entry["pto_values"].append(variant_row["pto_median_ms"])
                entry["baseline_ok"].append(variant_row["baseline_correct"])
                entry["pto_ok"].append(variant_row["pto_correct"])

        variants = []
        for entry in variant_map.values():
            baseline_median = statistics.median(entry["baseline_values"]) if entry["baseline_values"] else None
            pto_median = statistics.median(entry["pto_values"]) if entry["pto_values"] else None
            variants.append(
                {
                    "variant": entry["variant"],
                    "shape_summary": entry["shape_summary"],
                    "baseline_median_ms": baseline_median,
                    "pto_median_ms": pto_median,
                    "baseline_over_pto_pct": _ratio_pct(baseline_median, pto_median),
                    "all_correct": all(entry["baseline_ok"]) and all(entry["pto_ok"]),
                }
            )
        variants.sort(key=lambda item: _variant_key(item["variant"]))
        latest_report = json.loads(Path(rounds[-1]["latest_report_path"]).read_text(encoding="utf-8"))
        kernels.append(
            {
                "name": latest_report["name"],
                "family": latest_report["family"],
                "spec": spec_path,
                "latest_report_path": rounds[-1]["latest_report_path"],
                "rounds": rounds,
                "variants": variants,
            }
        )

    summary = {"generated_at": generated_at, "rounds": args.rounds, "kernels": kernels}
    json_path = REPORTS_DIR / f"regression_parity_{generated_at}.json"
    md_path = REPORTS_DIR / f"regression_parity_{generated_at}.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(md_path, summary)
    LATEST_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(LATEST_MD, summary)
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
