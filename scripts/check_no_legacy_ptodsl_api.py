#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [
    REPO_ROOT / "python",
    REPO_ROOT / "bench",
]

PATTERNS = {
    r"\btile\.": "legacy tile.* API",
    r"\bTileBufType\b": "legacy TileBufType public type",
    r"\bTileBufConfig\b": "legacy TileBufConfig public type",
    r"\bpto\.range\(": "legacy pto.range control-flow helper",
    r"\bpto\.cond\(": "legacy pto.cond helper",
    r"\bpto\.if_context\(": "legacy pto.if_context helper",
    r"\bpto\.vector_section\(": "legacy pto.vector_section helper",
    r"\bpto\.cube_section\(": "legacy pto.cube_section helper",
}


def scan() -> list[tuple[Path, int, str, str]]:
    findings: list[tuple[Path, int, str, str]] = []
    for root in SCAN_ROOTS:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for lineno, line in enumerate(text.splitlines(), start=1):
                for pattern, description in PATTERNS.items():
                    if re.search(pattern, line):
                        findings.append((path, lineno, description, line.strip()))
    return findings


def main() -> int:
    findings = scan()
    if not findings:
        print("No legacy PTODSL API usage found in pto-kernels.")
        return 0

    for path, lineno, description, line in findings:
        print(f"{path}:{lineno}: {description}: {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
