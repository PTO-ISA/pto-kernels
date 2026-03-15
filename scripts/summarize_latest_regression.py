#!/usr/bin/env python3
"""Summarize the latest generated NPU regression reports for active PTO kernels."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from pto_kernels.config import repo_root


REPO_ROOT = repo_root()
GENERATED_DIR = REPO_ROOT / "bench" / "generated"
REPORTS_DIR = REPO_ROOT / "bench" / "reports"
REGRESSION_LIST = REPO_ROOT / "bench" / "regression_kernels.yaml"
LATEST_JSON = REPORTS_DIR / "regression_latest.json"
LATEST_MD = REPORTS_DIR / "regression_latest.md"


def _load_specs() -> list[str]:
    data = yaml.safe_load(REGRESSION_LIST.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {REGRESSION_LIST}")
    return [str(item) for item in data]


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def _ratio_pct(baseline: float | None, pto: float | None) -> float | None:
    if baseline is None or pto is None or pto <= 0:
        return None
    return baseline / pto * 100.0


def _variant_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = report.get("baseline", {}).get("benchmark", {})
    pto = report.get("pto", {}).get("benchmark", {})
    baseline_limitations = baseline.get("baseline_limitations") or []
    baseline_variants = baseline.get("variant_reports") or []
    pto_variants = pto.get("variant_reports") or []
    pto_by_variant = {
        json.dumps(item.get("variant", {}), sort_keys=True): item for item in pto_variants if isinstance(item, dict)
    }

    def _passes(variant_report: dict[str, Any], benchmark: dict[str, Any]) -> bool:
        variant_correctness = variant_report.get("correctness") or {}
        benchmark_correctness = benchmark.get("correctness") or {}
        if "passes" in variant_correctness:
            return bool(variant_correctness["passes"])
        if "passes" in benchmark_correctness:
            return bool(benchmark_correctness["passes"])
        return variant_report.get("status", benchmark.get("status")) == "ok"

    rows: list[dict[str, Any]] = []
    for baseline_variant in baseline_variants:
        variant = baseline_variant.get("variant", {})
        key = json.dumps(variant, sort_keys=True)
        pto_variant = pto_by_variant.get(key, {})
        baseline_ms = baseline_variant.get("timings_ms", {}).get("median")
        pto_ms = pto_variant.get("timings_ms", {}).get("median")
        rows.append(
            {
                "variant": variant,
                "shape_summary": baseline_variant.get("shape_summary"),
                "baseline_status": baseline_variant.get("status", baseline.get("status")),
                "pto_status": pto_variant.get("status", pto.get("status")),
                "baseline_median_ms": baseline_ms,
                "pto_median_ms": pto_ms,
                "baseline_over_pto_pct": _ratio_pct(baseline_ms, pto_ms),
                "baseline_correct": _passes(baseline_variant, baseline),
                "pto_correct": _passes(pto_variant, pto),
                "baseline_limitations": baseline_limitations,
            }
        )
    if rows:
        return rows
    return [
        {
            "variant": {},
            "shape_summary": baseline.get("shape_summary"),
            "baseline_status": baseline.get("status"),
            "pto_status": pto.get("status"),
            "baseline_median_ms": baseline.get("timings_ms", {}).get("median"),
            "pto_median_ms": pto.get("timings_ms", {}).get("median"),
            "baseline_over_pto_pct": _ratio_pct(
                baseline.get("timings_ms", {}).get("median"),
                pto.get("timings_ms", {}).get("median"),
            ),
            "baseline_correct": _passes({}, baseline),
            "pto_correct": _passes({}, pto),
            "baseline_reason": baseline.get("reason"),
            "pto_reason": pto.get("reason"),
            "baseline_limitations": baseline_limitations,
        }
    ]


def _report_for_spec(spec_path: str) -> dict[str, Any]:
    spec = yaml.safe_load((REPO_ROOT / spec_path).read_text(encoding="utf-8"))
    report_path = GENERATED_DIR / spec["family"] / spec["name"] / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "name": spec["name"],
        "family": spec["family"],
        "spec": spec_path,
        "report_path": str(report_path),
        "variants": _variant_rows(report),
        "baseline_status": report.get("baseline", {}).get("benchmark", {}).get("status"),
        "pto_status": report.get("pto", {}).get("benchmark", {}).get("status"),
        "baseline_reason": report.get("baseline", {}).get("benchmark", {}).get("reason"),
        "pto_reason": report.get("pto", {}).get("benchmark", {}).get("reason"),
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Latest Regression Summary",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "| Kernel | Input Shape | Baseline | PTO | Baseline ms | PTO ms | Baseline / PTO | Correctness | Latest |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for kernel in summary["kernels"]:
        for variant in kernel["variants"]:
            correctness = "pass" if (variant["baseline_correct"] and variant["pto_correct"]) else "blocked/fail"
            lines.append(
                "| {name} | `{shape}` | {b_status} | {p_status} | {b_ms} | {p_ms} | {ratio} | {correctness} | [report]({report}) |".format(
                    name=kernel["name"],
                    shape=json.dumps(variant.get("shape_summary"), sort_keys=True),
                    b_status=variant.get("baseline_status", "n/a"),
                    p_status=variant.get("pto_status", "n/a"),
                    b_ms=_fmt_ms(variant.get("baseline_median_ms")),
                    p_ms=_fmt_ms(variant.get("pto_median_ms")),
                    ratio=_fmt_pct(variant.get("baseline_over_pto_pct")),
                    correctness=correctness,
                    report=Path(kernel["report_path"]).relative_to(REPO_ROOT),
                )
            )
            if variant.get("baseline_reason") or variant.get("pto_reason"):
                lines.append(
                    "|  |  | baseline reason: {b_reason} | pto reason: {p_reason} |  |  |  |  |  |".format(
                        b_reason=variant.get("baseline_reason", "n/a"),
                        p_reason=variant.get("pto_reason", "n/a"),
                    )
                )
            if variant.get("baseline_limitations"):
                lines.append(
                    "|  |  | baseline limitations: {limits} |  |  |  |  |  |  |".format(
                        limits="; ".join(str(item) for item in variant["baseline_limitations"]),
                    )
                )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    kernels = [_report_for_spec(spec_path) for spec_path in _load_specs()]
    summary = {
        "generated_at": datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ"),
        "kernels": kernels,
    }
    LATEST_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(LATEST_MD, summary)
    print(json.dumps({"latest_json": str(LATEST_JSON), "latest_md": str(LATEST_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
