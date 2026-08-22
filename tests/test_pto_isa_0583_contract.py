import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_pto_isa_0583.py"
SPEC = importlib.util.spec_from_file_location("check_pto_isa_0583", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def test_pto_isa_0583_repository_contract():
    assert CHECKER.check_repository(Path.cwd()) == []


def test_retired_body_branch_is_rejected_outside_legacy(tmp_path):
    (tmp_path / "PTO_ISA.lock.json").write_text(
        __import__("json").dumps(CHECKER.EXPECTED_LOCK), encoding="utf-8"
    )
    active = tmp_path / "benchmarks" / "supernpu"
    cube = active / "microbenchmark" / "cube"
    (cube / "src").mkdir(parents=True)
    for index in range(6):
        (cube / "src" / f"case_{index}.cpp").write_text(
            "constexpr int M = 1, N = 1, K = 1;\n", encoding="utf-8"
        )
    (cube / "cube_bench.hpp").write_text(
        "CubeTileM32 CubeTileN8 CubeAccumulatorM32 TLOAD_CUBE TSTORE_CUBE "
        "TMATMUL_ACC(tOut, tAcc, tA, tB)\n",
        encoding="utf-8",
    )
    (active / "microbenchmark" / "gen_cases.py").write_text("pass\n", encoding="utf-8")
    (active / "README.md").write_text(
        "PTO ISA 0.58.3 SizeCode 1..10 SizeCode 1..12 "
        "0000, 1000, 0100, 0010, 0001, 1100, 1110, 1111 "
        "TransA TransB CUBE_M16 CUBE_M32 CUBE_N8\n",
        encoding="utf-8",
    )
    (active / "bad.S").write_text("B.EQ target\n", encoding="utf-8")

    errors = CHECKER.check_repository(tmp_path)
    assert any("retired branch B.EQ" in error for error in errors)
