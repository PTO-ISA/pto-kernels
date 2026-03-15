#!/usr/bin/env python3
"""Check Python dependencies needed by local CANN TBE-side tooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from pto_kernels.utils.env import REQUIRED_TBE_PYTHON_MODULES, detect_env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    env = detect_env()
    modules = env.tbe_python_modules
    missing = [name for name in REQUIRED_TBE_PYTHON_MODULES if not modules.get(name, False)]

    payload = {
        "required_modules": list(REQUIRED_TBE_PYTHON_MODULES),
        "available_modules": modules,
        "missing_modules": missing,
        "toolkit_home": env.toolkit_home,
        "toolkit_version": env.toolkit_version,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"toolkit_home   : {env.toolkit_home}")
        print(f"toolkit_version: {env.toolkit_version}")
        print(f"missing        : {missing if missing else 'none'}")
        print(f"available      : {modules}")

    if args.strict and missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
