import importlib.util
from pathlib import Path
import shutil
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_pto_isa_0583.py"
SPEC = importlib.util.spec_from_file_location("check_pto_isa_0583", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)

GENERATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "supernpu"
    / "microbenchmark"
    / "gen_cases.py"
)
GENERATOR_SPEC = importlib.util.spec_from_file_location("gen_cases", GENERATOR_PATH)
assert GENERATOR_SPEC is not None and GENERATOR_SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(GENERATOR_SPEC)
sys.modules[GENERATOR_SPEC.name] = GENERATOR
GENERATOR_SPEC.loader.exec_module(GENERATOR)


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


def test_generated_microbenchmark_corpus_is_current():
    assert GENERATOR.check_generated(GENERATOR_PATH.parent) == []


def test_generated_microbenchmark_check_rejects_drift(tmp_path):
    source_root = GENERATOR_PATH.parent
    for family in ("vector", "memory", "cube", "scalar"):
        shutil.copytree(source_root / family, tmp_path / family)
    drift = tmp_path / "cube" / "src" / "tmatmul_fp32_32x32x32.cpp"
    drift.write_text(drift.read_text(encoding="utf-8") + "// drift\n", encoding="utf-8")
    errors = GENERATOR.check_generated(tmp_path)
    assert "cube: generated drift in src/tmatmul_fp32_32x32x32.cpp" in errors


def test_cube_accumulator_and_bias_types_are_architectural():
    micro_root = GENERATOR_PATH.parent
    header = (micro_root / "cube" / "cube_bench.hpp").read_text(encoding="utf-8")
    fp16_bias = (
        micro_root / "cube" / "src" / "tmatmul_bias_fp16_32x64x64.cpp"
    ).read_text(encoding="utf-8")
    assert "std::is_floating_point_v<D>, float" in header
    assert "std::is_signed_v<D>, int32_t, uint32_t" in header
    assert "gmC_t<AccD, 1, N>" in header
    assert "float bias[1*N];" in fp16_bias
    assert "bench_matmul_bias<__half,M,N,K>(c,a,b,bias);" in fp16_bias
