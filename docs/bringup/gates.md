# PTO 910B Bring-Up Gates

This document defines the canonical gate model for the PTO 910B bring-up
superproject.

A kernel should not advance status based on anecdotes. Status changes should be
explainable in terms of the gates below.

## Gate categories

### G0 — Workspace / environment gate

Purpose:
- confirm the current machine matches the intended `910B1 / ascend910b / a3 / dav-2201` target
- ensure local sibling repos and toolchain paths are discoverable

Minimum evidence:
- `bash scripts/source_env.sh`
- `python3 scripts/check_env.py --strict`

Pass expectation:
- `ptoas`, `bisheng`, and `torch_npu` are discoverable
- detected NPU maps to `ascend910b / a3 / dav-2201`

### G1 — Repo hygiene gate

Purpose:
- keep result interpretation trustworthy
- avoid accidental status changes from stray artifacts

Minimum evidence:
- `git status --short`
- explicit acknowledgement when build/test/result artifacts are intentionally tracked

Pass expectation:
- working tree is understood before running closure gates
- no accidental artifact churn is mixed with functional changes

### G2 — PTO compile gate

Purpose:
- verify the PTO path can lower through PTODSL -> PTOAS -> Bisheng for the bounded slice

Minimum evidence:
- PTO kernel compile path runs through the benchmark adapter or trace harness
- key artifacts are captured when available (`kernel.pto`, `kernel.cpp`, `.so`, caller sources)

Pass expectation:
- compile succeeds for the intended bounded slice
- or a blocked compile result is captured with a named blocker and artifact path

### G3 — Baseline contract gate

Purpose:
- confirm the baseline path exists for the operator slice
- when host runtime entrypoints do not exist, record the bounded reference contract explicitly

Minimum evidence:
- baseline adapter exists
- runtime entrypoint or explicit CPU/reference contract is recorded

Pass expectation:
- either a runnable baseline exists
- or a justified bounded reference path exists and the blocker is recorded against runtime/ops-transformer

### G4 — Correctness gate

Purpose:
- verify the PTO slice matches the intended contract for the bounded shape/layout/dtype scope

Minimum evidence:
- report with tolerance settings and max-abs-diff or equivalent metric
- clearly scoped shape/layout contract

Pass expectation:
- correctness passes for the declared bounded slice
- if not, the failure is reproducible and recorded

### G5 — Benchmark gate

Purpose:
- ensure the bounded slice is benchmarkable and not just a one-off compile artifact

Minimum evidence:
- repeated timing report
- warmup/repeat configuration
- output report persisted under `bench/results` or generated reports

Pass expectation:
- PTO benchmark runs reproducibly for the bounded slice
- baseline timing is recorded when available

### G6 — Regression gate

Purpose:
- include stable slices in the active regression matrix

Minimum evidence:
- entry in `bench/regression_kernels.yaml`
- latest summary generation path works

Pass expectation:
- kernel appears in the latest regression summary with interpretable state

### G7 — Cross-stack blocker gate

Purpose:
- keep ownership of PTODSL / PTOAS / PTO-ISA / runtime blockers explicit

Minimum evidence:
- blocker entry in `bench/gap_board.yaml`
- owning component and affected kernels are listed

Pass expectation:
- every incomplete kernel has an explainable blocker mapping
- mitigated/resolved blockers correspond to real evidence, not guesswork

### G8 — Performance gate

Purpose:
- converge toward useful performance after correctness and regression stability exist

Minimum evidence:
- PTO timing and baseline timing (when available)
- explicit statement of current gap and likely owner if PTO is slower

Pass expectation:
- performance is either acceptable for the target slice
- or the remaining gap is explicitly tracked as PTODSL/PTOAS/PTO-ISA/kernel debt

## Recommended closure ladder for a kernel

A bounded kernel slice should usually advance in this order:

1. G0 workspace/environment
2. G2 PTO compile
3. G3 baseline/reference contract
4. G4 correctness
5. G5 benchmark
6. G6 regression inclusion
7. G8 performance convergence

G1 and G7 apply throughout the entire process.

## Notes on blocked slices

A blocked slice is still useful if it is bounded and reproducible.
The bar for a “good blocked slice” is:

- benchmark spec exists
- adapters exist or are intentionally stubbed
- failure is reproducible
- owning blocker is named
- the next engineer does not have to rediscover the same failure manually

## Notes on performance

A kernel is not considered complete merely because:

- it compiles,
- it runs once, or
- it passes correctness for one shape.

Performance closure is a required final bring-up dimension. However, during
seed-kernel bring-up the acceptable intermediate state is:

- correctness-green,
- benchmarkable,
- regression-visible,
- performance debt explicitly tracked.
