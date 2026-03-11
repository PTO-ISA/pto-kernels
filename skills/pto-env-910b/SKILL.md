---
name: pto-env-910b
description: Use when working on PTO bring-up or kernel migration on the local 910B environment. Detect the NPU model, source the Ascend toolkit, verify ptoas and bisheng, and confirm the correct ascend910b/a3/dav-2201 target tuple.
---

# PTO Env 910B

Use this skill before building or benchmarking PTO kernels.

Workflow:

1. Run `bash scripts/source_env.sh`.
2. Run `python3 scripts/check_env.py --strict`.
3. Read `external/manifest.lock` to confirm pinned upstream commits.
4. If `ptoas` is missing, use `scripts/bootstrap_workspace.sh` and prefer the pinned `external/src/PTOAS` checkout.
5. Treat `910B1 -> ascend910b -> a3 -> dav-2201` as the required target mapping for this repo.

Important files:

- `scripts/source_env.sh`
- `scripts/check_env.py`
- `external/manifest.lock`
