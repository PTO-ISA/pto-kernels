---
name: pto-env-910b
description: Use when working on PTO bring-up or kernel migration on the local 910B environment. Detect the NPU model, source the Ascend toolkit, verify ptoas and bisheng, and confirm the correct ascend910b/a3/dav-2201 target tuple.
---

# PTO Env 910B

Use this skill before building or benchmarking PTO kernels.

Workflow:

1. Run `bash scripts/source_env.sh`.
2. Run `python3 scripts/check_env.py --strict`.
3. Confirm `ASCEND_TOOLKIT_HOME` resolves to the active CANN 9.x install and that the custom `ops-transformer` runtime is on `PYTHONPATH`.
4. Read `external/manifest.lock` to confirm pinned upstream commits.
5. If `ptoas` is missing, use `scripts/bootstrap_workspace.sh` and prefer the pinned `external/src/PTOAS` checkout.
6. Treat `910B1 -> ascend910b -> a3 -> dav-2201` as the required target mapping for this repo.
7. Keep PTO source sync-free at the DSL level and rely on `enable_insert_sync=True` / `ptoas` autosync for pipeline synchronization.

Important files:

- `scripts/source_env.sh`
- `scripts/check_env.py`
- `external/manifest.lock`
