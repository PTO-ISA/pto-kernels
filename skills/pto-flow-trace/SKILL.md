---
name: pto-flow-trace
description: Use when tracing or debugging the PTO-DSL to PTOAS to pto-isa compilation flow. Preserve kernel.pto, generated C++, compile commands, and failure attribution across ptodsl, PTOAS, and pto-isa.
---

# PTO Flow Trace

Use this skill when a kernel fails to compile or when you need preserved artifacts.

Workflow:

1. Source the environment with `bash scripts/source_env.sh`.
2. Use `python3 scripts/trace_flow.py <kernel.py> --output-dir <dir>` to inspect the PTO flow.
3. If the kernel exposes `build_jit_wrapper()`, rerun with `--build`.
4. Attribute failures in this order:
   `ptodsl` surface and metadata, then `PTOAS` lowering and legality, then `pto-isa` backend coverage.
5. Update `bench/gap_board.yaml` with the blocker and affected kernels.

Important files:

- `scripts/trace_flow.py`
- `bench/gap_board.yaml`
- `bench/canonical_compile_flags.yaml`
