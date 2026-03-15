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
3. Prefer `python3 scripts/tune_seed_kernels.py` for seed kernels with exposed env knobs; use the generated report in `bench/reports/` to choose the best PTO configuration before final parity runs.
4. Run repeated default-config parity through `python3 scripts/measure_seed_parity.py --rounds 3` when you need low-variation reporting on the current checked-in kernel defaults.
5. Use `python3 scripts/measure_regression_parity.py --rounds 1` for the checked-in regression matrix in `bench/regression_kernels.yaml` when you need a full sweep and the host is quiet enough for a long run.
6. Prefer `python3 scripts/summarize_latest_regression.py` when a fresh full sweep would be too noisy or too expensive; it rebuilds `bench/reports/regression_latest.md` and `bench/reports/regression_latest.json` from the stable `bench/generated/` tree.
7. Run the chosen single benchmark through `python3 -m pto_kernels.bench.runner <spec>` when debugging one kernel.
8. Use `bench/generated/<family>/<op>/` as the stable location for the latest generated `kernel.pto`, `kernel.cpp`, `caller.cpp`, `kernel.so`, and `report.json`.
9. Persist per-run history under `bench/results/` and summary reports under `bench/reports/`.
10. Record the PTO env knobs used for the winning run when a kernel exposes `pto_kernels.utils.tuning`.
11. Report performance as `baseline / PTO * 100%`, include the validated input shapes for each variant, and mark a kernel as `parity` only if PTO latency is within `10%` of the baseline on the same input and stream.
12. Treat `moe_distribute_dispatch` and `moe_distribute_combine` as special MC2 routing cases on this host: their latest reports are still baseline-blocked by the local 8-rank routing path, so use the bounded timeout reports in `bench/generated/mc2/` as the source of truth instead of forcing repeated unstable reruns.

Important files:

- `bench/specs/`
- `bench/regression_kernels.yaml`
- `bench/generated/`
- `bench/results/`
- `bench/reports/`
- `bench/canonical_compile_flags.yaml`
- `scripts/measure_seed_parity.py`
- `scripts/measure_regression_parity.py`
- `scripts/summarize_latest_regression.py`
- `scripts/tune_seed_kernels.py`
