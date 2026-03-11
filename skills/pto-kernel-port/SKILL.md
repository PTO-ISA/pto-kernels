---
name: pto-kernel-port
description: Use when porting one ops-transformer AI Core kernel into PTO-DSL. Follow the standard layout under python/pto_kernels/ops, add metadata and benchmark specs, preserve artifacts, and update kernel status through the shared checklist.
---

# PTO Kernel Port

Use this skill for one kernel at a time.

Workflow:

1. Create or update `python/pto_kernels/ops/<family>/<op>/kernel.py`.
2. Keep shapes, dtypes, tiling, wave, and blockers in `meta.py`.
3. Add or update `bench/specs/<family>/<op>.yaml`.
4. Ensure both adapters exist under `bench/adapters/ops_transformer` and `bench/adapters/ptodsl`.
5. Preserve `kernel.pto`, `kernel.cpp`, and `.so` artifacts with `scripts/trace_flow.py`.
6. Update `checklists/phase1_seed_kernels.md` or `checklists/910b_ai_core_migration.md` and the gap board.

Important files:

- `python/pto_kernels/ops/`
- `bench/specs/`
- `bench/adapters/`
- `checklists/`
