#!/usr/bin/env python3
"""Minimal patch(1) fallback for the ops-transformer Abseil patch step."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


@dataclass
class FilePatch:
    path: str
    hunks: list[Hunk]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-p", dest="strip", type=int, default=0)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("paths", nargs="*")
    return parser.parse_args()


def _strip_path(path: str, count: int) -> str:
    parts = Path(path).parts
    stripped = parts[count:]
    return str(Path(*stripped)) if stripped else path


def _parse_patch(text: str) -> list[FilePatch]:
    lines = text.replace("\r\n", "\n").splitlines()
    patches: list[FilePatch] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line.startswith("diff --git "):
            idx += 1
            continue
        idx += 1
        old_path = None
        new_path = None
        hunks: list[Hunk] = []
        while idx < len(lines):
            line = lines[idx]
            if line.startswith("diff --git "):
                break
            if line.startswith("--- "):
                old_path = line[4:].split("\t", 1)[0]
                idx += 1
                continue
            if line.startswith("+++ "):
                new_path = line[4:].split("\t", 1)[0]
                idx += 1
                continue
            match = HUNK_RE.match(line)
            if not match:
                idx += 1
                continue
            idx += 1
            hunk_lines: list[str] = []
            while idx < len(lines):
                current = lines[idx]
                if current.startswith("diff --git ") or HUNK_RE.match(current):
                    break
                if current.startswith(("\\",)):
                    idx += 1
                    continue
                hunk_lines.append(current)
                idx += 1
            hunks.append(
                Hunk(
                    old_start=int(match.group("old_start")),
                    old_count=int(match.group("old_count") or "1"),
                    new_start=int(match.group("new_start")),
                    new_count=int(match.group("new_count") or "1"),
                    lines=hunk_lines,
                )
            )
        if not old_path or not new_path or not hunks:
            raise ValueError("Unsupported patch format.")
        target = new_path if new_path != "/dev/null" else old_path
        patches.append(FilePatch(path=target, hunks=hunks))
    return patches


def _apply_hunks(target: Path, hunks: list[Hunk]) -> None:
    original = target.read_text(encoding="utf-8")
    has_trailing_newline = original.endswith("\n")
    content = original.splitlines()
    offset = 0
    for hunk in hunks:
        start = hunk.old_start - 1 + offset
        cursor = start
        replacement: list[str] = []
        original_slice: list[str] = []
        for hunk_line in hunk.lines:
            prefix = hunk_line[:1]
            payload = hunk_line[1:]
            if prefix == " ":
                if cursor >= len(content) or content[cursor] != payload:
                    raise ValueError(f"Context mismatch in {target} at line {cursor + 1}.")
                replacement.append(payload)
                original_slice.append(payload)
                cursor += 1
                continue
            if prefix == "-":
                if cursor >= len(content) or content[cursor] != payload:
                    raise ValueError(f"Delete mismatch in {target} at line {cursor + 1}.")
                original_slice.append(payload)
                cursor += 1
                continue
            if prefix == "+":
                replacement.append(payload)
                continue
            raise ValueError(f"Unsupported hunk line prefix: {prefix!r}")
        content[start:cursor] = replacement
        offset += len(replacement) - len(original_slice)
    updated = "\n".join(content)
    if has_trailing_newline or updated:
        updated += "\n"
    target.write_text(updated, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if args.help:
        print("patch shim: supports `patch -pN < file.patch`.")
        return 0
    if args.paths:
        print("patch shim only supports stdin input.", file=sys.stderr)
        return 2
    patch_text = sys.stdin.read()
    if not patch_text.strip():
        return 0
    try:
        patches = _parse_patch(patch_text)
        for file_patch in patches:
            target = Path.cwd() / _strip_path(file_patch.path, args.strip)
            _apply_hunks(target, file_patch.hunks)
    except Exception as exc:  # pragma: no cover - shell-facing fallback
        print(f"patch shim failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
