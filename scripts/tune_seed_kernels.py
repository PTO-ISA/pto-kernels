#!/usr/bin/env python3
"""Sweep safe tuning knobs for phase-1 PTO seed kernels on 910B."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from pto_kernels.bench.runner import BenchmarkRunner
from pto_kernels.config import repo_root


REPO_ROOT = repo_root()
REPORTS_DIR = REPO_ROOT / "bench" / "reports"
LATEST_JSON = REPORTS_DIR / "kernel_parity_latest.json"
LATEST_MD = REPORTS_DIR / "kernel_parity_latest.md"
SEED_CASES = (
    {
        "name": "grouped_matmul",
        "spec": "bench/specs/gmm/grouped_matmul.yaml",
        "env_grid": {"PTO_GROUPED_MATMUL_BASE_K": (32, 64)},
    },
    {
        "name": "apply_rotary_pos_emb",
        "spec": "bench/specs/posembedding/apply_rotary_pos_emb.yaml",
        "env_grid": {"PTO_APPLY_ROTARY_BLOCK_DIM": (1, 2, 4, 8)},
    },
    {
        "name": "ffn",
        "spec": "bench/specs/ffn/ffn.yaml",
        "env_grid": {
            "PTO_FFN_BASE_K1": (32, 64),
            "PTO_FFN_BASE_K2": (32, 64),
        },
    },
    {
        "name": "moe_token_permute",
        "spec": "bench/specs/moe/moe_token_permute.yaml",
        "env_grid": {},
    },
    {
        "name": "flash_attention_score",
        "spec": "bench/specs/attention/flash_attention_score.yaml",
        "env_grid": {"PTO_ATTENTION_QK_BASE_K": (32, 64)},
    },
    {
        "name": "matmul_reduce_scatter",
        "spec": "bench/specs/mc2/matmul_reduce_scatter.yaml",
        "env_grid": {"PTO_MC2_BASE_K": (32, 64)},
    },
)


@contextmanager
def overridden_env(values: dict[str, str]) -> Iterator[None]:
    previous: dict[str, str | None] = {}
    try:
        for key, value in values.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key in values:
            old_value = previous.get(key)
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def iter_configs(env_grid: dict[str, tuple[int, ...]]) -> Iterator[dict[str, str]]:
    if not env_grid:
        yield {}
        return
    keys = list(env_grid)
    values = [env_grid[key] for key in keys]
    for combo in itertools.product(*values):
        yield {key: str(value) for key, value in zip(keys, combo, strict=True)}


def status_ok(report: dict, side: str) -> bool:
    benchmark = report.get(side, {}).get("benchmark", {})
    correctness = benchmark.get("correctness", {})
    return benchmark.get("status") == "ok" and bool(correctness.get("passes"))


def median_ms(report: dict, side: str) -> float | None:
    benchmark = report.get(side, {}).get("benchmark", {})
    timings = benchmark.get("timings_ms", {})
    value = timings.get("median")
    return float(value) if isinstance(value, (int, float)) else None


def baseline_ratio(baseline_ms: float | None, pto_ms: float | None) -> float | None:
    if baseline_ms is None or pto_ms is None or baseline_ms <= 0:
        return None
    return pto_ms / baseline_ms


def delta_fraction(baseline_ms: float | None, pto_ms: float | None) -> float | None:
    ratio = baseline_ratio(baseline_ms, pto_ms)
    if ratio is None:
        return None
    return ratio - 1.0


def aggregate_rounds(rounds: list[dict]) -> dict:
    baseline_values = [round_info["baseline_median_ms"] for round_info in rounds if round_info["baseline_median_ms"] is not None]
    pto_values = [round_info["pto_median_ms"] for round_info in rounds if round_info["pto_median_ms"] is not None]
    all_rounds_measured = all(round_info["status"] == "measured" for round_info in rounds)
    baseline_ms = statistics.median(baseline_values) if baseline_values else None
    pto_ms = statistics.median(pto_values) if pto_values else None
    return {
        "status": "measured" if all_rounds_measured and baseline_ms is not None and pto_ms is not None else "blocked",
        "baseline_median_ms": baseline_ms,
        "pto_median_ms": pto_ms,
        "pto_over_baseline": baseline_ratio(baseline_ms, pto_ms),
        "delta_over_baseline": delta_fraction(baseline_ms, pto_ms),
    }


def select_best(trials: list[dict]) -> dict | None:
    eligible = [trial for trial in trials if trial["aggregate"]["status"] == "measured"]
    if not eligible:
        return None
    return min(eligible, key=lambda trial: trial["aggregate"]["pto_median_ms"])


def summarize_round(report: dict, round_index: int) -> dict:
    baseline_ok = status_ok(report, "baseline")
    pto_ok = status_ok(report, "pto")
    baseline_ms = median_ms(report, "baseline")
    pto_ms = median_ms(report, "pto")
    return {
        "round": round_index,
        "status": "measured" if baseline_ok and pto_ok and baseline_ms is not None and pto_ms is not None else "blocked",
        "report_path": report["report_path"],
        "latest_report_path": report.get("latest_report_path"),
        "artifacts_dir": report["artifacts_dir"],
        "latest_artifacts_dir": report.get("latest_artifacts_dir"),
        "baseline_status": report.get("baseline", {}).get("benchmark", {}).get("status"),
        "pto_status": report.get("pto", {}).get("benchmark", {}).get("status"),
        "baseline_median_ms": baseline_ms,
        "pto_median_ms": pto_ms,
        "pto_over_baseline": baseline_ratio(baseline_ms, pto_ms),
        "delta_over_baseline": delta_fraction(baseline_ms, pto_ms),
        "baseline_reason": report.get("baseline", {}).get("benchmark", {}).get("reason"),
        "pto_reason": report.get("pto", {}).get("benchmark", {}).get("reason"),
        "baseline_correct": baseline_ok,
        "pto_correct": pto_ok,
    }


def summarize_trial(case_name: str, env_overrides: dict[str, str], round_reports: list[dict]) -> dict:
    return {
        "case": case_name,
        "env": env_overrides,
        "rounds": round_reports,
        "aggregate": aggregate_rounds(round_reports),
        "latest_report_path": round_reports[-1].get("latest_report_path") if round_reports else None,
        "latest_artifacts_dir": round_reports[-1].get("latest_artifacts_dir") if round_reports else None,
    }


def ratio_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}x"


def median_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Kernel Parity Summary",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Rounds per config: `{summary['rounds']}`",
        "",
        "| Kernel | Best env | Baseline ms | PTO ms | PTO / baseline | Status | Latest |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for case in summary["cases"]:
        best = case["best_trial"]
        if best is None:
            lines.append(f"| {case['name']} | n/a | n/a | n/a | n/a | blocked | n/a |")
            continue
        env = ", ".join(f"{k}={v}" for k, v in best["env"].items()) or "default"
        aggregate = best["aggregate"]
        latest = Path(best["latest_report_path"]).relative_to(REPO_ROOT) if best["latest_report_path"] else "n/a"
        lines.append(
            "| {name} | {env} | {baseline} | {pto} | {ratio} | {status} | [report]({latest}) |".format(
                name=case["name"],
                env=env,
                baseline=median_text(aggregate["baseline_median_ms"]),
                pto=median_text(aggregate["pto_median_ms"]),
                ratio=ratio_text(aggregate["pto_over_baseline"]),
                status=aggregate["status"],
                latest=latest,
            )
        )

    lines.extend(["", "## Trials", ""])
    for case in summary["cases"]:
        lines.append(f"### {case['name']}")
        lines.append("")
        for trial in case["trials"]:
            env = ", ".join(f"{k}={v}" for k, v in trial["env"].items()) or "default"
            aggregate = trial["aggregate"]
            lines.append(
                "- `{env}`: baseline={baseline} ms, pto={pto} ms, ratio={ratio}, status={status}, latest=`{latest}`".format(
                    env=env,
                    baseline=median_text(aggregate["baseline_median_ms"]),
                    pto=median_text(aggregate["pto_median_ms"]),
                    ratio=ratio_text(aggregate["pto_over_baseline"]),
                    status=aggregate["status"],
                    latest=(
                        Path(trial["latest_report_path"]).relative_to(REPO_ROOT)
                        if trial["latest_report_path"]
                        else "n/a"
                    ),
                )
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    runner = BenchmarkRunner()
    generated_at = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    cases: list[dict] = []

    for case in SEED_CASES:
        trials = []
        for env_overrides in iter_configs(case["env_grid"]):
            round_reports = []
            for round_index in range(1, args.rounds + 1):
                with overridden_env(env_overrides):
                    report = runner.run(case["spec"])
                round_reports.append(summarize_round(report, round_index))
            trials.append(summarize_trial(case["name"], env_overrides, round_reports))
        cases.append(
            {
                "name": case["name"],
                "spec": case["spec"],
                "trials": trials,
                "best_trial": select_best(trials),
            }
        )

    summary = {"generated_at": generated_at, "rounds": args.rounds, "cases": cases}
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
