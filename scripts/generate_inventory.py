#!/usr/bin/env python3
"""Generate the 910B kernel inventory from ops-transformer."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


def load_inventory(ops_transformer_root: Path) -> dict[str, object]:
    op_list = ops_transformer_root / "docs" / "zh" / "op_list.md"
    text = op_list.read_text(encoding="utf-8")
    entries = re.findall(
        r'<td><a href="\.\./\.\./([^/]+/[^/]+)/README\.md">([^<]+)</a></td>',
        text,
    )

    included = []
    ai_cpu = []
    a3_only = []

    for rel, name in entries:
        op_dir = ops_transformer_root / rel
        cml = op_dir / "op_host" / "CMakeLists.txt"
        content = cml.read_text(encoding="utf-8", errors="ignore") if cml.exists() else ""
        has_a2 = "ascend910b" in content
        has_a3 = "ascend910_93" in content
        record = {
            "family": rel.split("/")[0],
            "name": name,
            "ops_transformer_path": rel,
        }
        if not (op_dir / "op_kernel").exists():
            ai_cpu.append(record)
        elif has_a3 and not has_a2:
            a3_only.append(record)
        else:
            included.append(record)

    return {"included": included, "excluded": {"ai_cpu": ai_cpu, "a3_only": a3_only}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ops_transformer_root")
    parser.add_argument("--output")
    args = parser.parse_args()

    data = load_inventory(Path(args.ops_transformer_root).resolve())
    rendered = yaml.safe_dump(data, sort_keys=False)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
