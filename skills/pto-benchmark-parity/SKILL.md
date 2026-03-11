---
name: pto-benchmark-parity
description: Use when comparing a PTO-DSL kernel against the ops-transformer baseline on the local 910B machine. Run side-by-side correctness and latency checks with the shared benchmark protocol and update parity status.
---

# PTO Benchmark Parity

Use this skill after both baseline and PTO adapters exist.

Workflow:

1. Load the spec from `bench/specs/<family>/<op>.yaml`.
2. Use the fixed benchmark protocol from `bench/canonical_compile_flags.yaml`:
   20 warmups, 100 timed iterations, median latency.
3. Run the benchmark through `python3 -m pto_kernels.bench.runner <spec>`.
4. Persist the JSON report under `bench/results/`.
5. Mark the kernel as `parity` only if PTO latency is within `10%` of the baseline on the same input and stream.

Important files:

- `bench/specs/`
- `bench/results/`
- `bench/canonical_compile_flags.yaml`
