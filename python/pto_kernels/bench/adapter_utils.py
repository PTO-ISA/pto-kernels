"""Shared helpers for benchmark adapters."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def blocked(reason: str) -> dict[str, Any]:
    return {"status": "blocked", "reason": reason}


def describe_baseline(repo_root: Path, family: str, name: str, ops_transformer_path: str) -> dict[str, Any]:
    return {
        "family": family,
        "name": name,
        "ops_transformer_path": ops_transformer_path,
        "baseline_root": str(repo_root.parent / "ops-transformer"),
    }


def describe_pto(repo_root: Path, kernel_path: str, meta_path: str) -> dict[str, Any]:
    meta_file = repo_root / meta_path
    kernel_file = repo_root / kernel_path
    result = {
        "kernel_path": str(kernel_file),
        "meta_path": str(meta_file),
    }
    if meta_file.exists():
        module = load_module(meta_file)
        result["meta"] = getattr(module, "META", {})
    return result


def compile_pto_kernel(repo_root: Path, kernel_path: str, output_dir: Path) -> dict[str, Any]:
    kernel_file = repo_root / kernel_path
    module = load_module(kernel_file)
    builder = getattr(module, "build_jit_wrapper", None)
    if not callable(builder):
        return blocked("kernel module does not expose build_jit_wrapper(output_dir)")
    try:
        builder(output_dir=output_dir)
    except NotImplementedError as exc:
        return blocked(str(exc))
    return {"status": "ready", "kernel_path": str(kernel_file), "output_dir": str(output_dir)}
