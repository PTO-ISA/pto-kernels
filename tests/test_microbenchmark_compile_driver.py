from pathlib import Path
import os
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
MICRO_ROOT = REPO_ROOT / "benchmarks" / "supernpu" / "microbenchmark"
DRIVER = MICRO_ROOT / "compile_all.sh"
CUBE_DRIVER = MICRO_ROOT / "cube" / "compile.all"
COMPLETION = "Microbench build completed: all selected categories passed"


def run_script(script: Path, *, cwd: Path, env: dict[str, str]):
    return subprocess.run(
        ["/bin/bash", str(script)],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def mock_make(tmp_path: Path, return_code: int = 0) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make = bin_dir / "make"
    make.write_text(f"#!/bin/bash\nexit {return_code}\n", encoding="utf-8")
    make.chmod(0o755)
    return bin_dir


def test_cube_driver_fails_when_make_frontend_is_missing(tmp_path):
    env = os.environ.copy()
    env["PATH"] = str(tmp_path / "empty-bin")
    result = run_script(CUBE_DRIVER, cwd=CUBE_DRIVER.parent, env=env)
    assert result.returncode != 0
    assert "make: command not found" in result.stdout
    assert "=== cube FAILED:" in result.stdout
    assert "cube completed" not in result.stdout


def test_cube_driver_reports_success_with_mock_make(tmp_path):
    env = os.environ.copy()
    env["PATH"] = f"{mock_make(tmp_path)}:/usr/bin:/bin"
    result = run_script(CUBE_DRIVER, cwd=CUBE_DRIVER.parent, env=env)
    assert result.returncode == 0, result.stdout
    assert "=== cube completed: all 6 cases passed ===" in result.stdout
    assert "cube FAILED" not in result.stdout


def test_parent_aggregates_category_failure_without_completion(tmp_path):
    micro_root = tmp_path / "microbenchmark"
    for category in ("cube", "vector", "memory", "scalar"):
        category_root = micro_root / category
        category_root.mkdir(parents=True)
        return_code = 9 if category == "vector" else 0
        (category_root / "compile.all").write_text(
            f"#!/bin/bash\nexit {return_code}\n", encoding="utf-8"
        )
    env = os.environ.copy()
    env["MICROBENCH_ROOT"] = str(micro_root)
    result = subprocess.run(
        ["/bin/bash", str(DRIVER), "all"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode != 0
    assert "FAIL: vector category" in result.stdout
    assert "PASS: scalar category" in result.stdout
    assert "Microbench build FAILED: vector" in result.stdout
    assert COMPLETION not in result.stdout


def test_parent_cube_success_with_mock_frontend(tmp_path):
    env = os.environ.copy()
    env["PATH"] = f"{mock_make(tmp_path)}:/usr/bin:/bin"
    result = subprocess.run(
        ["/bin/bash", str(DRIVER), "cube"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert f"{COMPLETION} (cube)" in result.stdout
    assert "Microbench build FAILED" not in result.stdout
