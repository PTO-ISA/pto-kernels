# pto-isa kernels

Ascend NPU kernels using [pto-isa](https://github.com/PTO-ISA/pto-isa/). Parallel Tile Operation (PTO) is a virtual instruction set architecture designed by Ascend CANN, focusing on tile-level operations.

This repository now also hosts the PTO 910B bring-up superproject for migrating
`ops-transformer` AI Core kernels to the `PTO-DSL -> PTOAS -> pto-isa` flow on
the current `910B1` environment.

## Build

```bash
bash scripts/source_env.sh
pip3 install -r requirements.txt
make build_wheel
```

The above commands will generate a wheel (i.e., `pto_kernels-0.1.0-*.whl`) that is pip installable.

### Installing

```bash
pip install --force-reinstall pto_isa_kernels-*.whl
```

```bash
make test
```

## PTO 910B Bring-Up

Key assets:

- `external/manifest.lock`: pinned upstreams for `pto-dsl`, `PTOAS`, `pto-isa`, and `ops-transformer`
- `scripts/bootstrap_workspace.sh`: clone pinned sibling repos into `external/src`
- `scripts/check_env.py`: validate the local `910B1 -> ascend910b -> a3 -> dav-2201` toolchain
- `scripts/trace_flow.py`: preserve `kernel.pto`, `kernel.cpp`, and compiled artifacts for a PTO-DSL kernel
- `bench/`: benchmark specs, adapters, inventory, and gap tracking
- `skills/`: PTO Codex skills plus `scripts/install_codex_skills.sh`

Quick start:

```bash
bash scripts/source_env.sh
make check-env
make bootstrap
PYTHONPATH=python python3 scripts/check_env.py --json
```

## Tutorial

If you are new to this repository, start with the Chinese tutorial under
[`tutorial/`](./tutorial/README.md). It explains the full
`PTO-DSL -> PTOAS -> PTO-ISA -> Bisheng -> .so -> benchmark` workflow,
shows minimal PTODSL examples, and walks through real kernels such as
`grouped_matmul`, `flash_attention_score`, and `moe_token_permute` with
current performance data from `bench/reports/regression_latest.md`.
