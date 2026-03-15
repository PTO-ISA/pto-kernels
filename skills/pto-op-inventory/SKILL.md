---
name: pto-op-inventory
description: Use when classifying ops-transformer kernels for PTO migration. Generate or update the 910B inventory, assign waves, maintain seed-kernel lists, and keep backlog items separated from in-scope kernels.
---

# PTO Op Inventory

Use this skill when adding kernels to the migration program.

Workflow:

1. Treat `bench/kernel_inventory.yaml` as the current source of truth.
2. When the upstream ops list changes, regenerate candidate inventory data with `python3 scripts/generate_inventory.py ../ops-transformer`.
3. Keep AI CPU scheduler ops and A3-only MC2 ops under `excluded`.
4. Maintain `seed_kernels`, `wave`, `phase`, and `ops_transformer_path` for every in-scope kernel.
5. Mirror high-level execution order in `checklists/910b_ai_core_migration.md`.
6. Keep `bench/regression_kernels.yaml` aligned with the currently runnable 910B parity set.
7. Use `bench/reports/regression_latest.md` as the fast summary of what is green today:
   11 kernels are currently green on NPU, while `moe_distribute_dispatch` and `moe_distribute_combine` remain blocked by the local 8-rank MC2 routing baseline.

Important files:

- `bench/kernel_inventory.yaml`
- `bench/regression_kernels.yaml`
- `bench/reports/regression_latest.md`
- `checklists/910b_ai_core_migration.md`
- `scripts/generate_inventory.py`
