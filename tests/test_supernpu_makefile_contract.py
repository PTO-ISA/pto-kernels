import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCALAR_DIR = REPO_ROOT / "benchmarks" / "supernpu" / "microbenchmark" / "scalar"
MULTI_THREAD_VEC_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "supernpu"
    / "benchmark"
    / "one-level-arch"
    / "test"
    / "kernel"
    / "multi_thread"
    / "vec"
)
CHIP_DEFINITION = (
    REPO_ROOT
    / "benchmarks"
    / "supernpu"
    / "benchmark"
    / "one-level-arch"
    / "test"
    / "common"
    / "src"
    / "chip_def.h"
)
STARTUP_CALLS = (
    (CHIP_DEFINITION.parents[1] / "_start.s", "main"),
    (CHIP_DEFINITION.parent / "benchmark_boot_linx.s", "_linx_start"),
)


class SuperNpuMakefileContractTest(unittest.TestCase):
    def run_make(
        self,
        *extra: str,
        cwd: Path = SCALAR_DIR,
        testcase: str = "add_i32_lat",
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "make",
                "-B",
                "-n",
                f"TESTCASE={testcase}",
                "COMPILER_DIR=/tmp/linx-toolchain/bin",
                "LINX_TILEOP_API_ROOT=/tmp/linx-tileop-api",
                *extra,
                "diss",
            ],
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_linx_build_requires_explicit_sysroot(self) -> None:
        result = self.run_make()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LINX_SYSROOT is not set", result.stdout)

    def test_linx_build_applies_sysroot_to_compile_and_link(self) -> None:
        result = self.run_make("LINX_SYSROOT=/tmp/linx-sysroot")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--sysroot=/tmp/linx-sysroot", result.stdout)
        self.assertIn("-isystem /tmp/linx-sysroot/include/c++/v1", result.stdout)
        self.assertIn("-stdlib=libc++", result.stdout)

    def test_one_level_build_uses_the_same_explicit_sysroot_contract(self) -> None:
        result = self.run_make(
            "LINX_SYSROOT=/tmp/linx-sysroot",
            cwd=MULTI_THREAD_VEC_DIR,
            testcase="tadd",
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--target=linx64-unknown-linux-musl", result.stdout)
        self.assertIn("--sysroot=/tmp/linx-sysroot", result.stdout)
        self.assertIn("-isystem /tmp/linx-sysroot/include/c++/v1", result.stdout)
        self.assertIn("-stdlib=libc++", result.stdout)
        kernel_compile = next(
            line for line in result.stdout.splitlines() if "/src/tadd.cpp" in line
        )
        self.assertIn("-nostdinc++", kernel_compile)
        self.assertIn("-isystem /tmp/linx-sysroot/include/c++/v1", kernel_compile)

    def test_baremetal_heap_can_be_bounded_by_the_runtime_gate(self) -> None:
        chip_definition = CHIP_DEFINITION.read_text(encoding="utf-8")
        self.assertIn("#ifndef HEAP_SIZE", chip_definition)
        self.assertIn("#define HEAP_SIZE 0x1000000000", chip_definition)

        result = self.run_make(
            "LINX_SYSROOT=/tmp/linx-sysroot",
            "baremetal=on",
            "LINX_BAREMETAL_HEAP_SIZE=0x10000000",
            cwd=MULTI_THREAD_VEC_DIR,
            testcase="tadd",
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("-DHEAP_SIZE=0x10000000", result.stdout)

    def test_startup_call_blocks_are_explicitly_terminated(self) -> None:
        for source, target in STARTUP_CALLS:
            with self.subTest(source=source.name):
                text = source.read_text(encoding="utf-8")
                self.assertIn(f"HL.BSTART.STD CALL, {target}, ra=_end", text)
                self.assertIn("C.BSTOP\n_end:", text)


if __name__ == "__main__":
    unittest.main()
