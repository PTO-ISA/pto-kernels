#!/usr/bin/env python3
"""Lightweight fail-closed checks for the maintained SuperNPU v0.58 subtree."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EXPECTED_SOURCE = {
    "schema": "pto-kernels.supernpu-source.v1",
    "source_repo": "https://github.com/PTO-ISA/SuperNPUBench",
    "source_commit": "4d8fcb5d4f3ff845e0400da70a1b290094880493",
    "source_tree": "0f5fc1dff18db3abb91bcf02f22241bc01272be8",
    "import_path": "benchmarks/supernpu",
}

EXPECTED_MICROBENCH_COUNTS = {
    "vector": 123,
    "memory": 25,
    "cube": 6,
    "scalar": 124,
}

TEXT_SUFFIXES = {
    "",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".json",
    ".md",
    ".mk",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}

FORBIDDEN_ACTIVE_PATTERNS = {
    "retired B.IOD spelling": re.compile(r"\bB\.IOD\b"),
    "retired BSTART.PAR spelling": re.compile(r"\bBSTART\.PAR\b"),
    "retired C.B.IOS spelling": re.compile(r"\bC\.B\.IOS\b"),
    "old TMA block classification": re.compile(r"\bBSTART\.TMA\b"),
    "old TEPL classification": re.compile(r"\bTEPL\b|_TEPL\b|\bBSTART\.TEPL\b"),
    "retired Tile load/store spelling": re.compile(r"\bTCOPY(?:IN|OUT)\b"),
    "retired operation spelling": re.compile(
        r"\b(?:ACCCVT|TADDC|TADDSC|TAXPY|TFMOD|TFMODS|TGATHERB|TLRELU|"
        r"TPRELU|TRANDOM|TRESHAPE|TSORT32|TSUBC|TSUBSC)\b"
    ),
    "obsolete compiler target": re.compile(r"\blinx64v5\b"),
    "obsolete compiler option": re.compile(r"-mlxbc|enable-all-vector-as-tilereg"),
    "embedded legacy assembly implementation": re.compile(r"template_asm(?:\.h|\.hpp)"),
    "private machine path": re.compile(r"/remote/"),
}


def is_legacy_path(path: Path, active_root: Path) -> bool:
    relative = path.relative_to(active_root)
    parts = relative.parts
    return len(parts) >= 2 and parts[0] == "status" and parts[1] == "legacy"


def find_forbidden_active_terms(active_root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(active_root.rglob("*")):
        if not path.is_file() or is_legacy_path(path, active_root):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "Makefile":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in FORBIDDEN_ACTIVE_PATTERNS.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                errors.append(
                    f"{path.relative_to(active_root)}:{line}: {label}: {match.group(0)}"
                )
    return errors


def compile_all_cases(path: Path) -> set[str]:
    pattern = re.compile(r"\bmake\s+TESTCASE=([A-Za-z0-9_]+)")
    return set(pattern.findall(path.read_text(encoding="utf-8")))


def check_repository(repo_root: Path) -> list[str]:
    errors: list[str] = []
    active_root = repo_root / "benchmarks" / "supernpu"
    required = [
        active_root / "README.md",
        active_root / "SOURCE.lock.json",
        active_root / "docs" / "README.md",
        active_root / "status" / "legacy" / "README.md",
        active_root / "benchmark" / "one-level-arch",
        active_root / "microbenchmark",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing required path: {path.relative_to(repo_root)}")
    if errors:
        return errors

    lock = json.loads((active_root / "SOURCE.lock.json").read_text(encoding="utf-8"))
    for key, expected in EXPECTED_SOURCE.items():
        if lock.get(key) != expected:
            errors.append(
                f"SOURCE.lock.json {key!r}: expected {expected!r}, got {lock.get(key)!r}"
            )

    readme = (active_root / "README.md").read_text(encoding="utf-8")
    for engine in ("VEC", "TLSU", "CUBE", "SFU"):
        if engine not in readme:
            errors.append(f"active README is missing {engine} engine classification")

    if (active_root / "benchmark" / "two-level-arch").exists():
        errors.append(
            "embedded two-level API remains active; move it under status/legacy"
        )

    embedded_api = list((active_root / "benchmark").rglob("template_asm.hpp"))
    embedded_api = [
        path for path in embedded_api if not is_legacy_path(path, active_root)
    ]
    if embedded_api:
        errors.extend(
            f"active embedded assembly authority: {path.relative_to(active_root)}"
            for path in embedded_api
        )

    errors.extend(find_forbidden_active_terms(active_root))

    micro_root = active_root / "microbenchmark"
    for family, expected_count in EXPECTED_MICROBENCH_COUNTS.items():
        source_dir = micro_root / family / "src"
        actual_cases = {path.stem for path in source_dir.glob("*.cpp")}
        if len(actual_cases) != expected_count:
            errors.append(
                f"{family} microbench count: expected {expected_count}, got {len(actual_cases)}"
            )
        listed_cases = compile_all_cases(micro_root / family / "compile.all")
        missing = sorted(actual_cases - listed_cases)
        extra = sorted(listed_cases - actual_cases)
        if missing or extra:
            errors.append(
                f"{family} compile.all mismatch: missing={missing[:5]} extra={extra[:5]}"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="pto-kernels repository root",
    )
    args = parser.parse_args()
    errors = check_repository(args.root.resolve())
    if errors:
        print("SuperNPU v0.58 checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    total = sum(EXPECTED_MICROBENCH_COUNTS.values())
    print(f"SuperNPU v0.58 checks passed: {total} microbench cases; legacy excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
