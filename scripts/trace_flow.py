#!/usr/bin/env python3
"""Trace the PTO-DSL -> PTOAS -> bisheng flow for a kernel module."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from pto_kernels.utils import detect_env


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_module", help="Path to kernel.py exposing build_jit_wrapper().")
    parser.add_argument("--output-dir", required=True, help="Directory for generated artifacts.")
    parser.add_argument("--build", action="store_true", help="Compile artifacts with the JIT wrapper.")
    args = parser.parse_args()

    module_path = Path(args.kernel_module).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = detect_env()
    module = _load_module(module_path)

    report: dict[str, object] = {
        "kernel_module": str(module_path),
        "output_dir": str(output_dir),
        "environment": json.loads(env.to_json()),
    }

    builder = getattr(module, "build_jit_wrapper", None)
    if not callable(builder):
        report["status"] = "blocked"
        report["reason"] = "kernel module does not expose build_jit_wrapper(output_dir)"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    try:
        wrapper = builder(output_dir=output_dir)
        report["wrapper_type"] = type(wrapper).__name__
        report["artifact_paths"] = getattr(wrapper, "_artifact_paths", lambda: ())()
        if args.build:
            getattr(wrapper, "_build")()
            report["status"] = "built"
        else:
            report["status"] = "described"
    except NotImplementedError as exc:
        report["status"] = "blocked"
        report["reason"] = str(exc)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
