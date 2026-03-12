# PTO 910B Bring-Up

This repository is the superproject for migrating `ops-transformer` AI Core
kernels to the `PTO-DSL -> PTOAS -> pto-isa` flow on the local `910B1`
environment.

The current phase-1 PTO kernel sources are manual-sync-free. They no longer
emit DSL-level event record/wait pairs and instead rely on `ptoas` sync
insertion on the `ascend910b` / `a3` path.

## Current Target

- NPU family: `910B1`
- SoC target: `ascend910b`
- PTO backend: `--pto-arch=a3`
- Bisheng backend: `--npu-arch=dav-2201`
- CANN version: `9.0.0-beta.1`

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

## First Executable Slices

The first runnable PTO slices are currently:

### `grouped_matmul`

This seed is a deliberately constrained dense variant:

- single batch
- one dense weight
- BF16 inputs
- F32 accumulation with BF16 output
- no bias, activation, quantization, or routing

This variant exists to validate the PTO-DSL -> PTOAS -> bisheng -> `.so`
pipeline on the current `910B1` host before the full grouped semantics land.

Current verified state on this host:

- baseline median latency: about `0.115 ms`
- PTO median latency: about `0.274 ms`
- correctness: both paths pass at `atol=1e-3` with current max abs diff
  about `4.88e-4`
- the PTO seed now stores the ACC tile directly to a BF16 GM tensor view, so
  the first seed no longer depends on a separate BF16 epilogue helper
- the PTO kernel source no longer carries manual event/wait pairs; generated
  sync now comes from `ptoas`

Current known bring-up blockers:

- `ptodsl` still needs reusable routing/group-list primitives for the full
  grouped GEMM semantics.
- The PTO seed is correct on the baseline BF16 contract now, but it is still
  slower than the installed grouped-matmul runtime on the current seed shape,
  so remaining work is performance tuning and semantic expansion rather than
  output-type closure.

### `apply_rotary_pos_emb`

This seed now runs end-to-end in both baseline and PTO paths for two constrained
half-mode rope variants:

- `layout = TND`, shape `64 x 1 x 128`
- `layout = BSND`, shape `2 x 32 x 1 x 128`
- `rotary_mode = half`
- `dtype = float16`
- `query_heads = key_heads = 1`
- `head_dim = 128`

Current benchmark on this host:

- baseline median latency: about `0.107 ms`
- PTO median latency: about `0.218 ms`
- correctness: exact match to the fp16 half-precision reference for both paths
- the PTO rope kernel now also relies on `ptoas`-inserted sync instead of
  manual event wiring

The active remaining rope gap is now generalization rather than basic bring-up:
PTODSL now covers the first TND and BSND half-mode seed shapes, but it still
needs broader rotary-mode support and reusable posembedding helpers for the
rest of the rope family.

### `ffn`

The dense FFN seed now runs end-to-end in both baseline and PTO paths for a
constrained phase-1 slice:

- shape: `M=32`, `K=128`, `N1=256`, `N2=128`
- dtype: `float16`
- activation: `relu`
- no bias
- no expert tokens
- seed tensors scaled by `0.125` to keep the fp16 reference contract stable

Current benchmark on this host:

- baseline median latency: about `0.095 ms`
- PTO median latency: about `0.438 ms`
- correctness: both paths pass at `atol=1e-3`, and the current PTO seed lands
  an exact match to the fp16 reference contract after output rounding

The PTO seed currently runs as a staged pipeline:

1. cube matmul `x @ w1 -> hidden`
2. vector relu in place on `hidden`
3. cube matmul `hidden @ w2 -> out`

That staged pipeline is now factored through reusable helpers in
`python/pto_kernels/ops/ffn/common.py`. FFN is no longer blocked on basic
staged execution, but it does not close the full FFN-family migration yet. The
active remaining gap is fused cube-vector-cube lowering in PTODSL/PTOAS so
later FFN-family kernels do not depend on explicit GM round-trips between
stages. The active PTO FFN source no longer emits manual event/wait pairs.

### `moe_token_permute`

The `moe_token_permute` seed now runs end-to-end in both baseline and PTO paths
for a constrained routing slice:

- tokens shape: `8 x 16`
- dtype: `float16`
- indices: 1D `int32`
- top-1 routing only
- `padded_mode = false`
- `num_out_tokens = 0`

Reference contract for this seed:

- `permuted_tokens = tokens[argsort(indices)]`
- `sorted_indices_out = inverse_permutation(argsort(indices))`

Current benchmark on this host:

- baseline median latency: about `0.107 ms`
- PTO median latency: about `0.342 ms`
- correctness: both paths pass with exact match on tokens and sorted indices

The PTO seed currently runs as a constrained two-stage path:

1. a PTO gather kernel reorders the flattened token buffer
2. a PTO copy kernel returns the routing metadata tensor for the current seed

The benchmark harness currently provides the flattened gather map for this
seed, so the active remaining MoE-routing gap is not basic permutation
execution anymore. It is reusable on-device sort, inverse-permutation, and
routed reorder/scatter support so later MoE ports do not depend on host-side
preprocessing.

### `flash_attention_score`

The constrained dense attention seed now runs end-to-end in both baseline and
PTO paths:

- shape: `B=1`, `N=1`, `S=32`, `D=64`
- layout: `BNSD`
- dtype: `float16`
- no masks, prefix, rope, or dropout
- `sparse_mode = 0`

Reference contract for this seed:

- plain scaled-dot-product attention with scale `1 / sqrt(D)`

Current benchmark on this host:

- baseline median latency: about `0.150 ms`
- PTO median latency: about `0.430 ms`
- correctness: both paths pass at `atol=1e-3`, current PTO max abs diff about
  `6.10e-5`

The PTO seed currently runs as a constrained staged path:

1. cube matmul `Q @ K^T -> scores`
2. vector row-wise softmax on the `32 x 32` score tile
3. cube matmul `P @ V -> out`

That staged dense-attention path is now factored through reusable helpers in
`python/pto_kernels/ops/attention/common.py`. The benchmark harness pre-scales
the PTO query tensor by `1 / sqrt(64)` so the seed matches the baseline
attention contract. The active remaining gap is no longer basic attention
bring-up. It is broader masked/online attention generalization plus PTO-side
performance closure.

### `matmul_reduce_scatter`

The final phase-1 seed is now classified with the real baseline contract rather
than a placeholder:

- shape: `M=128`, `K=256`, `N=128`
- dtype: `float16`
- contract: `reduce_scatter(x1 @ x2)`
- bias: disabled
- reduce op: `sum`
- first target world size: `2`

The installed baseline entrypoint on this host is
`torch_npu.npu_mm_reduce_scatter_base`, and it now runs through a repo-local
multi-rank HCCL harness on the current machine:

- local world size: `2`
- devices: `npu:0`, `npu:1`
- `hcom` acquired from `get_hccl_comm_name`
- measured baseline median latency: about `0.557 ms`
- correctness: passes at `atol=1e-3`, current max abs diff about `8.4e-4`
- reference contract: `reduce_scatter(sum_i(x1_i @ x2_i))`

The harness is intentionally local and seed-oriented for now, but it proves the
baseline runtime contract on `ascend910b` and can be reused for later MC2
adapter bring-up.

The constrained PTO side now runs as well for the same `world_size = 2` seed:

- PTO local contract: dense fp16 local matmul `x1 @ x2 -> local_mm`
- distributed contract: the benchmark harness performs `all_reduce(sum)` on
  `local_mm` and then row-chunks the reduced result per rank
- PTO median latency: about `0.701 ms`
- correctness: passes at `atol=1e-3`, current max abs diff about `8.4e-4`

This is intentionally not a true PTODSL MC2 collective kernel yet. It is a
host-orchestrated distributed seed that proves the local PTO matmul can plug
into the repo-local HCCL launcher and match the baseline contract on this
machine. The active remaining MC2 gap is reusable PTODSL collective surface and
non-ad hoc distributed launch support for later MC2 kernels.

## Baseline Bring-Up

Use the repo-local bring-up scripts for the seed package instead of ad hoc
commands:

```bash
source scripts/source_env.sh
python3 scripts/check_ops_transformer_runtime.py
bash scripts/bringup_ops_transformer_seeds.sh --install
python3 scripts/check_ops_transformer_runtime.py
```

Defaults:

- target SoC: `ascend910b`
- seed package ops: `apply_rotary_pos_emb,grouped_matmul,ffn,moe_token_permute,flash_attention_score,matmul_reduce_scatter`
- install root: inferred from `ASCEND_TOOLKIT_HOME`
- effective package path: auto-synthesized under `build/ops_transformer_cann_compat` when the local CANN install does not expose `share/info/*/version.info`
- build log: written to `build/logs/ops_transformer/bringup_<timestamp>.log`

Expected outcome for this phase:

- `build_out/cann-ops-transformer-custom_linux-aarch64.run` or `build_out/cann-*-ops-transformer_*.run` exists in the local workspace
- the package installs into the current toolkit root
- runtime probe reports installed package metadata before per-kernel smoke runs begin

Current local state:

- The metadata-layout blocker is mitigated. The PTO workspace can still
  synthesize a compatible package root when needed, but the current
  `9.0.0-beta.1` install already exposes the required `share/info/*/version.info`
  metadata directly.
- The open-package API gap is resolved on this host with CANN
  `9.0.0-beta.1`. The seed package now builds successfully and installs under
  `vendors/custom_transformer` in the active toolkit root.
- The `grouped_matmul` seed now runs end-to-end in both baseline and PTO paths
  on this machine. Current benchmark shape:
  - `M=128`, `K=128`, `N=128`
  - BF16 inputs
  - baseline runtime contract uses a single 3D weight tensor for
    `aclnnGroupedMatmulV5`
  - PTO path still emits `f32` outputs until the reusable BF16 epilogue lands
- The current result is a functioning side-by-side benchmark, not performance
  parity yet. The baseline is faster on this seed shape, so the active
  remaining closure work is PTO-side feature/performance tuning rather than
  environment bring-up.
- `matmul_reduce_scatter` is no longer the phase-1 exception. Both baseline and
  PTO seeds now run on the local 2-rank HCCL harness, with the PTO side still
  depending on a host-orchestrated collective contract rather than on real
  PTODSL MC2 collectives.

## Phase 1 Status

The six seed kernels are now fully classified:

- `grouped_matmul`: baseline and PTO both run
- `apply_rotary_pos_emb`: baseline and PTO both run
- `ffn`: baseline and PTO both run; PTO remains slower and still relies on a staged pipeline rather than a fused kernel
- `moe_token_permute`: baseline and PTO both run; PTO still depends on a host-precomputed gather map rather than on-device routing primitives
- `flash_attention_score`: baseline and PTO both run; PTO remains a staged dense seed and still needs masked/online attention generalization plus performance tuning
- `matmul_reduce_scatter`: baseline and PTO both run on the local 2-rank HCCL harness; PTO still depends on a host-orchestrated collective contract rather than on PTODSL MC2 collectives

The current phase-1 audit artifact is:

- JSON: `bench/reports/phase1_seed_audit.json`
- Markdown: `bench/reports/phase1_seed_audit.md`

That audit is green on this host: all six seeds have baseline and PTO reports
with successful correctness checks.

## Phase 2 Focus

Phase 2 is no longer about basic bring-up. The remaining blockers have been
narrowed to real reusable feature work:

- broader rope-mode generalization beyond the current TND and BSND half-mode seeds
- on-device MoE routing sort/inverse-permutation primitives
- masked/online attention softmax, cache, and sparse helpers beyond the reusable dense staged path
- true PTODSL MC2 collectives rather than host-orchestrated collectives
- fused cube/vector/cube lowering beyond the reusable staged FFN path

## Waves

- Wave 1: `posembedding`, `gmm`, `ffn`
- Wave 2: `moe`
- Wave 3: attention core
- Wave 4: attention advanced
- Wave 5: `ascend910b` MC2 subset
