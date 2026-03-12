#!/usr/bin/env python3
"""Fail if active PTO migration code still uses the legacy `scalar as s` API."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    REPO_ROOT / "python" / "pto_kernels" / "ops",
    REPO_ROOT.parent / "pto-dsl" / "tests" / "frontend",
)
PATTERNS = (
    re.compile(r"\bfrom ptodsl import scalar as s\b"),
    re.compile(r"\bs\.(?:const|index_cast|ceil_div|div_s|rem_s|min_u|eq|lt|gt|ge|select)\b"),
)


def main() -> int:
    violations: list[str] = []
    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines()
            for pattern in PATTERNS:
                for match in pattern.finditer(text):
                    line_no = text.count("\n", 0, match.start()) + 1
                    violations.append(f"{path}:{line_no}: {pattern.pattern}")
                    if len(violations) >= 200:
                        break
                if len(violations) >= 200:
                    break
            if len(violations) >= 200:
                break
        if len(violations) >= 200:
            break

    if violations:
        print("Legacy PTODSL scalar API usage detected:")
        for item in violations:
            print(item)
        return 1

    print("No legacy PTODSL scalar API usage found in active PTO migration sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
