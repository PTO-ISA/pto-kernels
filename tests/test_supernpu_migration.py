import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_supernpu_v058.py"
SPEC = importlib.util.spec_from_file_location("check_supernpu_v058", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def test_supernpu_v058_repository_contract():
    assert CHECKER.check_repository(Path.cwd()) == []


def test_legacy_is_excluded_but_active_retired_spelling_fails(tmp_path):
    legacy = tmp_path / "status" / "legacy"
    legacy.mkdir(parents=True)
    (legacy / "history.md").write_text("BSTART.TEPL\n", encoding="utf-8")
    assert CHECKER.is_legacy_path(legacy / "history.md", tmp_path)
    assert CHECKER.find_forbidden_active_terms(tmp_path) == []

    active = tmp_path / "benchmark"
    active.mkdir()
    (active / "bad.hpp").write_text('asm("B.IOD");\n', encoding="utf-8")
    errors = CHECKER.find_forbidden_active_terms(tmp_path)
    assert len(errors) == 1
    assert "retired B.IOD spelling" in errors[0]


def test_compile_all_inventory_accepts_guarded_run_case(tmp_path):
    compile_all = tmp_path / "compile.all"
    compile_all.write_text(
        "run_case first_case\nrun_case second_case\n", encoding="utf-8"
    )
    assert CHECKER.compile_all_cases(compile_all) == {"first_case", "second_case"}
