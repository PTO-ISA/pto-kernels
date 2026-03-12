---
name: pto-kernel-port
description: Use when porting one ops-transformer AI Core kernel into PTO-DSL. Follow the standard layout under python/pto_kernels/ops, add metadata and benchmark specs, preserve artifacts, and update kernel status through the shared checklist.
---

# PTO Kernel Port

Use this skill for one kernel at a time.

Workflow:

1. Create or update `python/pto_kernels/ops/<family>/<op>/kernel.py`.
2. Keep PTO kernels sync-free at the DSL level; use `enable_insert_sync=True` and let `ptoas` insert synchronization.
3. Keep shapes, dtypes, tiling, wave, and blockers in `meta.py`.
4. Add or update `bench/specs/<family>/<op>.yaml`.
5. Ensure both adapters exist under `bench/adapters/ops_transformer` and `bench/adapters/ptodsl`.
6. Expose reusable tuning knobs through `pto_kernels.utils.tuning` when a seed has safe search dimensions.
7. Preserve `kernel.pto`, `kernel.cpp`, and `.so` artifacts with `scripts/trace_flow.py`, and keep the latest generated files refreshed under `bench/generated/<family>/<op>/`.
8. Validate correctness and parity with repeated runs when the kernel is meant for reporting, not just a one-off smoke run.
9. Update `checklists/phase1_seed_kernels.md` or `checklists/910b_ai_core_migration.md` and the gap board.

Important files:

- `python/pto_kernels/ops/`
- `bench/specs/`
- `bench/adapters/`
- `bench/generated/`
- `python/pto_kernels/utils/tuning.py`
- `checklists/`
