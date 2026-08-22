#!/usr/bin/env python3
"""Fail-closed PTO ISA 0.58.3 authority and workload contract checks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_LOCK: dict[str, Any] = {
    "schema": "pto-kernels.pto-isa-lock.v1",
    "release": "0.58.3",
    "release_tag": "v0.58.3",
    "release_url": "https://github.com/PTO-ISA/pto-spec/releases/tag/v0.58.3",
    "source_commit": "e599a3d36ebfad43362ff591ea5e128816c684c7",
    "source_tree": "abb6899d2e664e378ac9c1b77062670daa4d31b4",
    "encoding_abi": "pto-isa-0.58.3-mode-function-v1",
    "encoding_projection_sha256": (
        "8a48b80e04484c70870f155bf9efc79d2a805cf99e809f4e4e8a7e6a7eb34172"
    ),
    "content_sha256": (
        "f299fe3d256c5d071e57bb4aaa2be2de2e4a386ae090048df1f73ae92d392678"
    ),
    "instruction_counts": {
        "scalar": 466,
        "command": 74,
        "tile": 109,
        "extension_reservations": 40,
    },
    "local_size_code_bytes": [
        None,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
    ],
    "shared_size_code_bytes": [
        None,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
    ],
    "pe_mode_masks": [0, 8, 4, 2, 1, 12, 14, 15],
    "cube_layout_codes": {
        "ND2M32": 21,
        "ND2M16": 22,
        "ND2N8": 23,
        "M322ND": 24,
        "M162ND": 25,
        "N82ND": 26,
    },
    "b_fpatr_tail_fields": ["TransA", "TransB"],
    "retired_body_branches": [
        "B.EQ",
        "B.NE",
        "B.LT",
        "B.GE",
        "B.LTU",
        "B.GEU",
        "B.Z",
        "B.NZ",
    ],
}


def _active_files(active_root: Path):
    for path in sorted(active_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(active_root)
        if relative.parts[:2] == ("status", "legacy"):
            continue
        yield path


def active_cube_inventory(active_root: Path) -> list[Path]:
    one_level = active_root / "benchmark" / "one-level-arch"
    trigger = re.compile(r"\b(?:TMATMUL|TGEMV)|CubeTile|CubeAccumulator")
    inventory: list[Path] = []
    for path in sorted(one_level.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {
            ".c",
            ".cc",
            ".cpp",
            ".h",
            ".hpp",
        }:
            continue
        if trigger.search(path.read_text(encoding="utf-8")):
            inventory.append(path)
    return inventory


def check_repository(repo_root: Path) -> list[str]:
    errors: list[str] = []
    lock_path = repo_root / "PTO_ISA.lock.json"
    if not lock_path.is_file():
        return ["missing PTO_ISA.lock.json"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock != EXPECTED_LOCK:
        errors.append("PTO_ISA.lock.json does not match the exact 0.58.3 authority")

    active_root = repo_root / "benchmarks" / "supernpu"
    retired = tuple(EXPECTED_LOCK["retired_body_branches"])
    retired_pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(item) for item in retired) + r")\b"
    )
    for path in _active_files(active_root):
        if path.suffix.lower() not in {".c", ".cc", ".cpp", ".h", ".hpp", ".s", ".S"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        match = retired_pattern.search(text)
        if match:
            errors.append(
                f"{path.relative_to(repo_root)} uses retired branch {match.group(0)}"
            )

    cube_header_path = active_root / "microbenchmark" / "cube" / "cube_bench.hpp"
    cube_header = cube_header_path.read_text(encoding="utf-8").split("#if 0", 1)[0]
    for required in (
        "CubeTileM32",
        "CubeTileM16",
        "CubeTileN8",
        "CubeAccumulatorM32",
        "TLOAD_CUBE",
        "TSTORE_CUBE",
        "TMATMUL_ACC(tOut, tAcc, tA, tB)",
        "Tile<Location::Vec, cube_accumulator_element_t<D>, 1, N,",
    ):
        if required not in cube_header:
            errors.append(f"cube_bench.hpp is missing {required}")
    for forbidden in ("TileLeft", "TileRight", "bench_matmul_mx"):
        if forbidden in cube_header:
            errors.append(f"active cube_bench.hpp retains {forbidden}")
    for required in (
        "cube_accumulator_element_t",
        "std::is_signed_v<D>",
        "int32_t, uint32_t",
        "gmC_t<AccD, 1, N>",
        "tBias_t<D, N>",
    ):
        if required not in cube_header:
            errors.append(f"cube_bench.hpp is missing accumulator trait {required}")

    cube_inventory = active_cube_inventory(active_root)
    if not cube_inventory:
        errors.append("active one-level CUBE inventory is empty")
    inventory_path = active_root / "CUBE_ACTIVE.json"
    if not inventory_path.is_file():
        errors.append("missing CUBE_ACTIVE.json")
    else:
        inventory_data = json.loads(inventory_path.read_text(encoding="utf-8"))
        recorded_sources = inventory_data.get("active_sources", [])
        discovered_sources = [
            path.relative_to(active_root).as_posix() for path in cube_inventory
        ]
        if inventory_data.get("schema") != "pto-kernels.supernpu-cube-active.v1":
            errors.append("CUBE_ACTIVE.json schema mismatch")
        if inventory_data.get("release") != "0.58.3":
            errors.append("CUBE_ACTIVE.json release mismatch")
        if recorded_sources != discovered_sources:
            errors.append(
                "CUBE_ACTIVE.json active_sources do not match discovered reachability"
            )
    for source in cube_inventory:
        text = source.read_text(encoding="utf-8")
        for required in (
            "CubeTileM16",
            "CubeTileM32",
            "CubeTileN8",
            "CubeAccumulatorM16",
            "CubeAccumulatorM32",
            "TLOAD_CUBE",
            "TSTORE_CUBE",
        ):
            if required not in text:
                errors.append(f"{source.relative_to(repo_root)} is missing {required}")
        for label, pattern in (
            ("TileLeft", r"\bTileLeft\s*<"),
            ("TileRight", r"\bTileRight\s*<"),
            ("TMATMUL_FIXP", r"\bTMATMUL_FIXP\s*\("),
            ("ordinary CUBE TLOAD", r"\bTLOAD\s*\("),
            ("ordinary CUBE TSTORE", r"\bTSTORE\s*\("),
            ("Vec accumulator/output", r"Tile\s*<\s*Location::Vec"),
        ):
            if re.search(pattern, text):
                errors.append(f"{source.relative_to(repo_root)} retains {label}")
        for call in re.findall(r"TMATMUL_ACC\s*\(([^)]*)\)", text):
            operands = [operand.strip() for operand in call.split(",")]
            if len(operands) < 4 or operands[0] == operands[1]:
                errors.append(
                    f"{source.relative_to(repo_root)} lacks distinct explicit ACC D/C"
                )
        for name, value in re.findall(r"#define\s+(tilM|kTm|tM)\s+(\d+)", text):
            if int(value) > 32:
                errors.append(
                    f"{source.relative_to(repo_root)} has illegal Local CUBE {name}={value}"
                )

    retired_fa = (
        active_root
        / "status"
        / "legacy"
        / "one-level-cube-v058-incomplete"
        / "fa_2d_unroll_gmma.cpp"
    )
    if not retired_fa.is_file():
        errors.append("incomplete FA GMMA sketch is not isolated under status/legacy")

    cube_sources = sorted(
        (active_root / "microbenchmark" / "cube" / "src").glob("*.cpp")
    )
    if len(cube_sources) != 6:
        errors.append(f"expected 6 generated CUBE cases, got {len(cube_sources)}")
    for source in cube_sources:
        text = source.read_text(encoding="utf-8")
        match = re.search(r"constexpr int M = (\d+), N = (\d+), K = (\d+);", text)
        if match is None:
            errors.append(
                f"{source.relative_to(repo_root)} has no static M/N/K contract"
            )
            continue
        if int(match.group(1)) > 32:
            errors.append(
                f"{source.relative_to(repo_root)} exceeds CUBE_M32 row capacity"
            )

    generator = (active_root / "microbenchmark" / "gen_cases.py").read_text(
        encoding="utf-8"
    )
    active_generator_lines = [
        line.strip()
        for line in generator.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if any('cube("TMATMUL_MX"' in line for line in active_generator_lines):
        errors.append(
            "generator enables TMATMUL_MX without an exact optional-scale contract"
        )

    docs = re.sub(
        r"\s+",
        " ",
        (active_root / "README.md").read_text(encoding="utf-8"),
    )
    for required in (
        "PTO ISA 0.58.3",
        "SizeCode 1..10",
        "SizeCode 1..12",
        "0000, 1000, 0100, 0010, 0001, 1100, 1110, 1111",
        "TransA",
        "TransB",
        "CUBE_M16",
        "CUBE_M32",
        "CUBE_N8",
    ):
        if required not in docs:
            errors.append(f"SuperNPU README is missing {required}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    errors = check_repository(args.root.resolve())
    if errors:
        print("PTO ISA 0.58.3 checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PTO ISA 0.58.3 authority and workload checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
