# pto-kernels

A collection of high-performance custom kernels for **Ascend NPUs**, built on top of
[pto-isa](https://github.com/PTO-ISA/pto-isa) — the Parallel Tile Operation virtual
instruction set architecture designed by Ascend CANN.

PTO focuses on tile-level operations, enabling efficient, composable kernel development
targeting Huawei's Ascend AI processors.

This repository also hosts the PTO 910B bring-up superproject for migrating
`ops-transformer` AI Core kernels to the `PTO-DSL -> PTOAS -> pto-isa` flow on the
current `910B1` environment.

---

## Prerequisites

- A configured **torch-npu** environment
- Ascend toolkit installed at `/usr/local/Ascend/ascend-toolkit`

Run the one-time setup before building:

```bash
make setup_once
```

## Install repository using pip

The repository is "pip installable", i.e.,

```bash
export CMAKE_GENERATOR="Unix Makefiles" && pip install -v git+https://github.com/huawei-csl/pto-kernels.git
```

---

## Build

```bash
bash scripts/source_env.sh
pip3 install -r requirements.txt
make build_wheel
```

This produces an installable Python wheel:

```text
pto_kernels-X.Y.Z-*.whl
```

---

## Installation

```bash
pip install --force-reinstall pto_kernels-*.whl
```

---

## Testing

```bash
make test
```

---

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

## Repository Structure

```text
pto-kernels/
├── csrc/                  # C++ kernel source files
├── python/pto_kernels/    # Python bindings and utilities
├── examples/jit_cpp/      # JIT compilation examples
├── tests/                 # Test suite
├── scripts/               # Helper scripts
├── doxygen/               # API documentation config
└── CMakeLists.txt         # CMake build configuration
```

## Tutorial

If you are new to this repository, start with the Chinese tutorial under
[`tutorial/`](./tutorial/README.md). It explains the full
`PTO-DSL -> PTOAS -> PTO-ISA -> Bisheng -> .so -> benchmark` workflow,
shows minimal PTODSL examples, and walks through real kernels such as
`grouped_matmul`, `flash_attention_score`, and `moe_token_permute` with
current performance data from `bench/reports/regression_latest.md`.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## License

BSD-3-Clause-Clear — see [LICENSE](LICENSE) for details.
