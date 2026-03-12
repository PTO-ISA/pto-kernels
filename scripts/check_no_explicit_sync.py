#!/usr/bin/env python3
"""Fail if PTO authoring layers still contain explicit event/wait helpers."""

from __future__ import annotations

import re
from pathlib import Path

from pto_kernels.config import repo_root


PATTERN = re.compile(r"\b(record_event|wait_event|record_wait_pair)\b")
SCAN_ROOTS = (
    "python/pto_kernels",
    "../pto-dsl/ptodsl",
    "../pto-dsl/examples",
    "../pto-dsl/tests",
)


def main() -> int:
    root = repo_root()
    violations: list[str] = []
    for rel_root in SCAN_ROOTS:
        scan_root = (root / rel_root).resolve()
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if PATTERN.search(line):
                    violations.append(f"{path}:{line_no}:{line.strip()}")
    if violations:
        print("Explicit sync helpers are still present:")
        for violation in violations:
            print(violation)
        return 1
    print("No explicit PTO sync helpers found in PTO authoring layers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
