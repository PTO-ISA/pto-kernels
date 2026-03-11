# PTO 910B Bring-Up

This repository is the superproject for migrating `ops-transformer` AI Core
kernels to the `PTO-DSL -> PTOAS -> pto-isa` flow on the local `910B1`
environment.

## Current Target

- NPU family: `910B1`
- SoC target: `ascend910b`
- PTO backend: `--pto-arch=a3`
- Bisheng backend: `--npu-arch=dav-2201`
- CANN version: `8.5.0.alpha002`

## Phase 0 Deliverables

1. Pin workspace dependencies in `external/manifest.lock`.
2. Clone pinned repos into `external/src` with `scripts/bootstrap_workspace.sh`.
3. Source CANN and PTO workspace variables via `scripts/source_env.sh`.
4. Validate `npu-smi`, CANN, `ptoas`, `bisheng`, and `torch_npu` via `scripts/check_env.py`.
5. Keep canonical benchmark and compile settings in `bench/canonical_compile_flags.yaml`.
6. Keep migration inventory and initial blockers in `bench/kernel_inventory.yaml` and `bench/gap_board.yaml`.
7. Sync PTO skills into Codex with `scripts/install_codex_skills.sh`.

## Phase 1 Seed Kernels

- `apply_rotary_pos_emb`
- `grouped_matmul`
- `ffn`
- `moe_token_permute`
- `flash_attention_score`
- `matmul_reduce_scatter`

For each seed kernel, the bring-up path is:

1. Create PTO-DSL source under `python/pto_kernels/ops/<family>/<op>/kernel.py`.
2. Define shapes, dtypes, tiling, launch config, and blockers in `meta.py`.
3. Keep a benchmark spec in `bench/specs/<family>/<op>.yaml`.
4. Use `bench/adapters/ops_transformer/...` for the baseline path.
5. Use `bench/adapters/ptodsl/...` for the PTO path.
6. Preserve `kernel.pto`, `kernel.cpp`, caller source, and `.so` artifacts with `scripts/trace_flow.py`.
7. Record correctness and latency results under `bench/results/`.

## First Executable Slice

The first runnable PTO slice is currently the `grouped_matmul` seed in a
deliberately constrained dense variant:

- single batch
- one dense weight
- BF16 inputs
- F32 accumulation/output
- no bias, activation, quantization, or routing

This variant exists to validate the PTO-DSL -> PTOAS -> bisheng -> `.so`
pipeline on the current `910B1` host before the full grouped semantics land.

Current known bring-up blockers:

- `ptodsl` still needs a reusable BF16 epilogue conversion path, so the PTO
  seed keeps the output in `float32`.
- `ptodsl` still needs reusable routing/group-list primitives for the full
  grouped GEMM semantics.
- The runtime baseline for several `ops-transformer` kernels is not yet
  installed into the local CANN runtime, so `torch_npu` entrypoints can fail
  until the matching packages are built and installed.

## Waves

- Wave 1: `posembedding`, `gmm`, `ffn`
- Wave 2: `moe`
- Wave 3: attention core
- Wave 4: attention advanced
- Wave 5: `ascend910b` MC2 subset
