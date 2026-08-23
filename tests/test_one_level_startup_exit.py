from pathlib import Path
import os
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
STARTUP = (
    ROOT
    / "benchmarks"
    / "supernpu"
    / "benchmark"
    / "one-level-arch"
    / "test"
    / "common"
    / "_start.s"
)


def test_active_startup_uses_linux_exit_group_abi():
    source = STARTUP.read_text(encoding="utf-8")
    tail = source.split("_end:", 1)[1]
    assert "addi zero, 0x5e, ->a7" in tail
    assert "->x1" not in tail
    assert "->a0" not in tail
    assert tail.index("->a7") < tail.index("acrc 1") < tail.index("C.BSTOP")
    assert "a0 keeps main's status" in tail


def test_active_startup_target_disassembly(tmp_path):
    clang_text = os.environ.get("LINX_CLANG")
    objdump_text = os.environ.get("LINX_OBJDUMP")
    if not clang_text or not objdump_text:
        pytest.skip("set LINX_CLANG and LINX_OBJDUMP for target disassembly")
    clang = Path(clang_text)
    objdump = Path(objdump_text)
    if not clang.is_file() or not objdump.is_file():
        pytest.fail("configured Linx target tools do not exist")

    obj = tmp_path / "one-level-start.o"
    subprocess.run(
        [
            str(clang),
            "-target",
            "linx64-unknown-linux-musl",
            "-c",
            str(STARTUP),
            "-o",
            str(obj),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    disassembly = subprocess.run(
        [str(objdump), "-d", str(obj)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    end = disassembly.split("<_end>:", 1)[1]
    assert re.search(r"addi\s+zero,\s*94,\s*->a7", end)
    assert "->x1" not in end
    assert re.search(r"\bacrc\b", end)
