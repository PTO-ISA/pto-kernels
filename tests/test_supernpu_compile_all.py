from pathlib import Path
import os
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "benchmarks"
    / "supernpu"
    / "benchmark"
    / "one-level-arch"
    / "compile_all.sh"
)
OPERATORS = (
    "matmul",
    "broadcast",
    "concat",
    "gather",
    "transpose",
    "element_wise/gelu",
    "reduction/reducemax_col",
    "reduction/reducemax_row",
    "reduction/reducesum_col",
    "reduction/reducesum_row",
    "control",
    "sort",
    "deepseek",
)
COMPLETION = "Full compilation completed: all active operators passed"


def make_fixture(
    tmp_path: Path, *, missing: str | None = None, failing: str | None = None
):
    root = tmp_path / "one-level-arch"
    for operator in OPERATORS:
        if operator == missing:
            continue
        operator_root = root / "test" / "kernel" / operator
        operator_root.mkdir(parents=True)
        return_code = 7 if operator == failing else 0
        (operator_root / "compile.all").write_text(
            f"#!/bin/bash\nexit {return_code}\n", encoding="utf-8"
        )
    return root


def run_fixture(root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(COMPILER_DIR="/mock/linx/bin", REPO_ROOT=str(root))
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_missing_active_operator_fails_without_completion(tmp_path):
    result = run_fixture(make_fixture(tmp_path, missing="sort"))
    assert result.returncode != 0
    assert "active operator directory not found" in result.stdout
    assert COMPLETION not in result.stdout
    assert "Full compilation FAILED: sort" in result.stdout


def test_failing_operator_is_aggregated_and_fails(tmp_path):
    result = run_fixture(make_fixture(tmp_path, failing="gather"))
    assert result.returncode != 0
    assert "FAIL: gather compilation" in result.stdout
    assert COMPLETION not in result.stdout
    assert "Full compilation FAILED: gather" in result.stdout


def test_complete_active_inventory_reports_success(tmp_path):
    result = run_fixture(make_fixture(tmp_path))
    assert result.returncode == 0, result.stdout
    assert COMPLETION in result.stdout
    assert "Full compilation FAILED" not in result.stdout
    assert "/test/kernel/fa" not in result.stdout
