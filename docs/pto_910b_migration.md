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
- PTO median latency: about `0.261 ms` on the first upstream-shaped tiling rewrite
- correctness: both paths pass at `atol=1e-3` with current max abs diff
  about `4.88e-4`
- the PTO seed now stores the ACC tile directly to a BF16 GM tensor view, so
  the first seed no longer depends on a separate BF16 epilogue helper
- the PTO kernel source no longer carries manual event/wait pairs; generated
  sync now comes from `ptoas`
- the PTO kernel now mirrors the upstream `ops-transformer` host tiling more
  closely by splitting into `baseM x baseN` basic blocks and using the same
  diagonal-vs-row-major block traversal policy as the AscendC grouped-matmul
  kernel for the current compile-time shape

Current known bring-up blockers:

- `ptodsl` still needs reusable routing/group-list primitives for the full
  grouped GEMM semantics.
- `ptodsl` still lacks a reusable source-level surface for the upstream async
  preload / double-buffer / callback pipeline used by grouped-matmul and
  related cube-heavy kernels. PTOAS autosync is sufficient for correctness, but
  it is not yet a source-level replacement for the full overlap strategy.
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
- PTO median latency: about `0.214 ms`
- correctness: exact match to the fp16 half-precision reference for both paths
- the PTO rope kernel now also relies on `ptoas`-inserted sync instead of
  manual event wiring
- the PTO seed now mirrors the upstream core split more closely for the
  validated `64`-row seed by assigning one contiguous row chunk per core on the
  `apply_rotary_pos_emb` block grid instead of using the earlier generic
  ceil-div row partition

The active remaining rope gap is now generalization rather than basic bring-up:
PTODSL now covers the first TND and BSND half-mode seed shapes, but it still
needs broader rotary-mode support and reusable posembedding helpers for the
rest of the rope family. It also does not yet expose a source-level equivalent
to the upstream vector queue / double-buffer overlap pattern used in the
AscendC rope kernel, so the current PTO rewrite matches the core split more
closely than it matches the full copy/compute pipeline.

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

- baseline median latency: about `0.096 ms`
- PTO median latency: about `0.461 ms` on the first upstream-shaped tiling rewrite
- correctness: both paths pass at `atol=1e-3`, and the current PTO seed lands
  an exact match to the fp16 reference contract after output rounding

The PTO seed currently runs as a staged pipeline:

1. cube matmul `x @ w1 -> hidden`
2. vector relu in place on `hidden`
3. cube matmul `hidden @ w2 -> out`

That staged pipeline is now factored through reusable helpers in
`python/pto_kernels/ops/ffn/common.py`, and the phase-2 rewrite now follows the
upstream `ops-transformer` host tiling more closely by splitting both matmul
stages into `baseM x baseN` output tiles and distributing those tiles across
cores. FFN is no longer blocked on basic staged execution, but it does not
close the full FFN-family migration yet. The active remaining gap is fused
cube-vector-cube lowering plus a reusable source-level preload / overlap
surface in PTODSL/PTOAS so later FFN-family kernels do not depend on explicit
GM round-trips between stages. The active PTO FFN source no longer emits manual
event/wait pairs.

### `moe_token_permute`

The `moe_token_permute` seed now runs end-to-end in both baseline and PTO paths
for a constrained routing slice:

- token shapes: `8 x 16`, `16 x 16`, `16 x 32`
- dtype: `float16`
- indices: 1D `int32`
- top-1 routing only
- `padded_mode = false`
- `num_out_tokens = 0`

Reference contract for this seed:

- `permuted_tokens = tokens[argsort(indices)]`
- `sorted_indices_out = inverse_permutation(argsort(indices))`

Current benchmark on this host:

- baseline median latency: about `0.109 ms`
- PTO median latency: about `0.348 ms`
- `baseline / PTO * 100`: about `31.48%`
- correctness: both paths pass with exact match on tokens and sorted indices

The PTO seed currently runs as a constrained two-stage path:

1. a PTO gather kernel reorders the flattened token buffer with contiguous
   per-core token-row ownership that mirrors the upstream copy split more
   closely for the validated seed shapes
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
- PTO median latency: about `0.437 ms` on the first upstream-shaped tiling rewrite
- correctness: both paths pass at `atol=2e-3`, current PTO max abs diff about
  `1.33e-3`

The PTO seed currently runs as a constrained staged path:

1. cube matmul `Q @ K^T -> scores`
2. vector row-wise softmax on the `32 x 32` score tile
3. cube matmul `P @ V -> out`

That staged dense-attention path is now factored through reusable helpers in
`python/pto_kernels/ops/attention/common.py`, and the phase-2 rewrite now
follows the upstream `ops-transformer` host tiling more closely by splitting
the dense QK stage over `S1 x S2` score tiles, splitting the row-softmax stage
over contiguous `S1` chunks, and splitting the PV stage over `S1 x D` output
tiles. The benchmark harness pre-scales the PTO query tensor by `1 / sqrt(64)`
so the seed matches the baseline attention contract. The active remaining gap
is no longer basic attention bring-up. It is broader masked or online
attention generalization, deeper overlap parity, and PTO-side performance
closure.

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
distributed baseline and reference contract for the first MC2 collective slice.

The constrained PTO side now runs as well for the same `world_size = 2` seed:

- PTO local contract: dense fp16 local matmul `x1 @ x2 -> local_mm`
- distributed contract: the benchmark harness performs `all_reduce(sum)` on
  `local_mm` and then row-chunks the reduced result per rank
- validated shapes: `128x256x128` and `64x256x128`
- PTO median latency: about `0.766 ms`
- `baseline / PTO * 100`: about `77.53%`
- correctness: passes at `atol=1e-3`, current max abs diff about `8.4e-4`

This is intentionally not a true PTODSL MC2 collective kernel yet. It is a
host-orchestrated distributed seed that proves the local PTO matmul can plug
into the repo-local HCCL launcher and match the baseline contract on this
machine. The phase-2 rewrite now follows the upstream MC2 rank-chunk traversal
more closely by iterating output row chunks in rank order instead of treating
the local matmul as one monolithic block. The active remaining MC2 gap is still
reusable PTODSL collective surface and non-ad hoc distributed launch support
for later MC2 kernels.

### `moe_distribute_dispatch`

The next wave-5 MC2 slice is now scaffolded for the real A2 / 910B contract
rather than a synthetic toy shape:

- world size: `8`
- hidden size: `7168`
- routing: EP-only, top-1
- dtype: `float16`
- quantization: disabled
- local expert count: `1` per rank

The current PTO seed is deliberately narrow. It only covers the local
destination-major send-buffer packing stage and leaves the actual EP
`all_to_all` plus metadata tensor ownership to the benchmark harness. That is
still useful because it forces the port onto the same A2 shape regime and the
same destination-major local send order as the upstream AscendC dispatch path.

Current verified state on this host:

- the PTO seed compiles end-to-end through `PTO-DSL -> PTOAS -> bisheng`
  for the validated `8 x 7168` slice
- stable latest artifacts are emitted under
  `bench/generated/mc2/moe_distribute_dispatch`
- the active PTO source is still explicit-sync-free and relies on PTOAS
  insert-sync lowering
- the baseline adapter is wired to `torch_npu.npu_moe_distribute_dispatch`
  for the real `epWorldSize=8` contract

Current known blockers:

- the shared launcher now emits per-rank stage traces for 8-rank MC2 runs.
  On this host, the baseline `moe_distribute_dispatch` path reaches the
  worker body after HCCL init and the new per-rank worker traces show that
  every rank stalls inside the second warmup
  `torch_npu.npu_moe_distribute_dispatch` call. The remaining baseline issue
  is therefore no longer a generic launcher timeout
- the PTO dispatch path exposed a concrete `all_to_all_single` split-contract
  bug during bring-up: the first prototype was sending flattened element-count
  splits instead of row-count splits. The runtime is now fixed to use row
  splits on the `tokens x hidden` buffers, but full end-to-end 8-rank parity
  is still blocked by the baseline-side stall above
- the current PTODSL pack kernel still relies on a host-precomputed flattened
  gather map, so reusable on-device routing and scatter primitives are still
  missing
- widening this seed beyond the validated `8 x 7168` shape immediately runs
  into vector-memory pressure with the current flattened-gather strategy, so
  later work should replace it with a more native row-pack surface rather than
  scaling the stopgap design

### `moe_distribute_combine`

The paired wave-5 MC2 combine slice is now scaffolded against the same real A2
/ 910B contract as dispatch:

- world size: `8`
- hidden size: `7168`
- routing: EP-only, top-1
- dtype: `float16`
- local expert count: `1` per rank

This first combine prototype is intentionally narrower than the upstream
kernel. The current PTO seed only covers the final local integration stage
after the reverse-route buffer has already been compacted by the host harness.

Current verified state on this host:

- the PTO seed compiles end-to-end through `PTO-DSL -> PTOAS -> bisheng`
  for the validated `8 x 7168` slice
- the PTO seed now uses `pto.tscatter` through the new PTODSL `tile.scatter`
  surface over chunk-local flattened indices, and a local single-rank smoke
  reconstructs the original `8 x 7168` buffer with `max_abs_diff = 0.0`
- stable latest artifacts are emitted under
  `bench/generated/mc2/moe_distribute_combine`
- the active PTO source is still explicit-sync-free and relies on PTOAS
  insert-sync lowering
- the baseline adapter is wired to `torch_npu.npu_moe_distribute_combine`
  for the real `epWorldSize=8` contract

Current known blockers:

- the local 8-rank HCCL routing path is still not trustworthy on this host:
  with `PTO_ENABLE_UNSTABLE_8RANK_MC2=1` and a bounded 60-second launcher
  timeout, both the baseline and PTO `moe_distribute_combine` benchmarks time
  out before any rank emits a report
- the current PTO seed still consumes a host-precompacted reverse-route buffer
  plus host-generated chunk-local scatter indices instead of implementing the
  distributed reverse communication path and routing metadata handling on
  device
- later combine-family ports still need reusable routing, inverse-permutation,
  and MC2 collective surface rather than more host preprocessing

### `all_gather_matmul`

The first phase-2 MC2 extension now runs end-to-end in both baseline and PTO
paths for a constrained `all_gather_matmul` slice:

- validated global shapes: `128x256x128` and `256x256x128`
- dtype: `float16`
- contract: `output = allgather(x1) @ x2`, `gather_out = allgather(x1)`
- bias: disabled
- `gather_index = 0`
- `gather_output = true`
- first target world size: `2`

The installed baseline entrypoint on this host is
`torch_npu.npu_all_gather_base_mm`, and it now runs through the same repo-local
2-rank HCCL harness used by the other MC2 seed:

- local world size: `2`
- devices: `npu:0`, `npu:1`
- `hcom` acquired from `get_hccl_comm_name`
- baseline median latency: about `0.574 ms`
- correctness: matches both the gathered tensor and the matmul output at
  `atol=1e-3`

The constrained PTO side now runs for the same local harness and keeps PTO
source sync-free:

- PTO collective contract: the benchmark harness performs the HCCL
  `all_gather` outside the PTO kernel
- PTO math contract: dense fp16 global matmul over the gathered tensor
- traversal policy: local-first wrapped rank-chunk traversal that mirrors the
  upstream AscendC kernel more closely than a monolithic global matmul launch
- PTO median latency: about `1.044 ms`
- `baseline / PTO * 100`: about `54.97%`
- active remaining gap: PTODSL still lacks a true MC2 all-gather collective
  surface and the upstream async preload / overlap strategy, so this slice is a
  host-orchestrated correctness and tiling port rather than full MC2 pipeline
  parity

### `grouped_mat_mul_all_reduce`

The next phase-2 MC2 extension now runs end-to-end in both baseline and PTO
paths for a constrained single-group all-reduce slice:

- validated shapes: `M=128,K_total=256,N=128` and `M=256,K_total=256,N=128`
- per-rank local shapes: `x_local=[M,128]`, `weight_local=[128,128]`
- dtype: `float16`
- group count: `1`
- bias: disabled
- world size: `2`

The upstream repo ships this kernel, but the installed custom package on this
host does not expose `aclnnGroupedMatMulAllReduce`. For the current seed, the
baseline uses the same distributed math contract with available runtime pieces:

- local compute: `torch_npu.npu_grouped_matmul`
- distributed contract: `dist.all_reduce(sum)` over the `[M, N]` local result
- baseline median latency: about `0.724 ms`
- correctness: matches the summed reference `sum_i(x_i @ weight_i)` at
  `atol=1e-3`

The constrained PTO side now runs for the same local harness and keeps PTO
source sync-free:

- PTO local kernel: one dense-group grouped matmul with upstream-shaped
  `splitM / numBlocksN` turn-based core traversal
- distributed contract: the benchmark harness performs HCCL `all_reduce(sum)`
  outside the PTO kernel
- PTO median latency: about `0.736 ms`
- `baseline / PTO * 100`: about `98.28%`
- active remaining gap: PTODSL still lacks a true MC2 all-reduce collective
  surface and the upstream communication/compute overlap pipeline, so this
  slice is a host-orchestrated correctness and tiling port rather than full MC2
  pipeline parity

### `matmul_all_reduce`

The next phase-2 MC2 extension now runs end-to-end in both baseline and PTO
paths for a constrained dense all-reduce slice:

- validated shapes: `M=128,K=256,N=128` and `M=256,K=256,N=128`
- per-rank local shapes: `x1_local=[M,256]`, `x2=[256,128]`
- dtype: `float16`
- bias: disabled
- `x3`: disabled
- world size: `2`

The baseline uses the installed runtime entrypoint on this host:

- local compute plus collective: `torch_npu.npu_mm_all_reduce_base`
- baseline median latency: about `0.651 ms`
- correctness: matches the summed reference `sum_i(x1_local_i @ x2)` at
  `atol=1e-3`

The constrained PTO side now runs for the same local harness and keeps PTO
source sync-free:

- PTO local kernel: dense matmul with upstream-shaped `splitM / numBlocksN`
  turn-based core traversal
- distributed contract: the benchmark harness performs HCCL `all_reduce(sum)`
  outside the PTO kernel
- PTO median latency: about `0.748 ms`
- `baseline / PTO * 100`: about `87.04%`
- active remaining gap: PTODSL still lacks a true MC2 all-reduce collective
  surface and the upstream communication/compute overlap pipeline, so this
  slice is a host-orchestrated correctness and tiling port rather than full MC2
  pipeline parity

### `matmul_all_reduce_add_rms_norm`

The next phase-2 MC2 extension now runs end-to-end in both baseline and PTO
paths for a constrained dense all-reduce plus add+rms_norm slice:

- validated shapes: `M=128,K=256,N=128` and `M=256,K=256,N=128`
- per-rank local shapes: `x1_local=[M,256]`, `x2=[256,128]`
- residual shape: `[M,128]`
- gamma shape: `[128]`
- dtype: `float16`
- bias: disabled
- `x3`: disabled
- world size: `2`

The baseline uses the installed runtime entrypoints on this host:

- local compute plus collective: `torch_npu.npu_mm_all_reduce_base`
- epilogue: `torch_npu.npu_add_rms_norm`
- baseline median latency: about `0.599 ms`
- correctness: `y` and `norm_out` both pass at `atol=5e-3`

The constrained PTO side now runs for the same local harness and keeps PTO
source sync-free:

- PTO local kernel: dense matmul with upstream-shaped `splitM / numBlocksN`
  turn-based core traversal
- PTO epilogue: separate PTODSL vector `add + rms_norm`
- distributed contract: the benchmark harness performs HCCL `all_reduce(sum)`
  outside the PTO kernel
- PTO median latency: about `0.916 ms`
- `baseline / PTO * 100`: about `65.38%`
- current correctness state: `y` matches at `atol=5e-3`, but `norm_out`
  still drifts to about `1.17e-2` max abs diff, so this slice is runnable but
  not correctness-green yet
- issue found while extending this slice: a direct float32 widening attempt for
  the vector epilogue is currently illegal on the backend because the A2/A3
  `pto-isa` vector `TLOAD/TSTORE` path requires GM and vector-tile dtypes to
  match, so mixed-precision vector IO needs backend support before this epilogue
  can be widened cleanly in PTO source

### `inplace_matmul_all_reduce_add_rms_norm`

The next phase-2 MC2 extension now runs end-to-end in both baseline and PTO
paths for the constrained inplace dense all-reduce plus add+rms_norm slice:

- validated shapes: `M=128,K=256,N=128` and `M=256,K=256,N=128`
- per-rank local shapes: `x1_local=[M,256]`, `x2=[256,128]`
- residual inout shape: `[M,128]`
- gamma shape: `[128]`
- dtype: `float16`
- bias: disabled
- `x3`: disabled
- world size: `2`

The baseline uses the installed split runtime contract on this host:

- local compute plus collective: `torch_npu.npu_mm_all_reduce_base`
- epilogue: `torch_npu.npu_add_rms_norm`
- inplace contract: the harness validates that `y` is written back to the
  residual buffer
- baseline median latency: about `0.656 ms`
- correctness: the residual-inplace `y` contract and `norm_out` both pass at
  `atol=5e-3`

The constrained PTO side now runs for the same local harness and keeps PTO
source sync-free:

- PTO local kernel: dense matmul with upstream-shaped `splitM / numBlocksN`
  turn-based core traversal
- PTO epilogue: separate PTODSL vector `add + rms_norm`
- inplace contract: currently modeled as separate PTO epilogue output plus host
  copy-back into the residual buffer because native same-buffer aliasing is not
  yet validated on this path
- distributed contract: the benchmark harness performs HCCL `all_reduce(sum)`
  outside the PTO kernel
- PTO median latency: about `1.020 ms`
- `baseline / PTO * 100`: about `64.34%`
- current correctness state: the residual-inplace `y` contract matches at
  `atol=5e-3`, but `norm_out` still drifts to about `1.17e-2` max abs diff, so
  this slice is runnable but not correctness-green yet

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
- `all_gather_matmul` now also runs on the local 2-rank HCCL harness in both
  baseline and PTO paths, with the PTO side still depending on a host-side
  all-gather contract rather than on PTODSL MC2 collectives.
- `grouped_mat_mul_all_reduce` now runs on the local 2-rank HCCL harness in
  both baseline and PTO paths for a constrained single-group seed, with the PTO
  side still depending on a host-side all-reduce contract rather than on PTODSL
  MC2 collectives.
- `matmul_all_reduce` now runs on the local 2-rank HCCL harness in both
  baseline and PTO paths for a constrained dense seed, with the PTO side still
  depending on a host-side all-reduce contract rather than on PTODSL MC2
  collectives.
- `matmul_all_reduce_add_rms_norm` now runs on the local 2-rank HCCL harness in
  both baseline and PTO paths for a constrained dense seed, with the PTO side
  still depending on a host-side all-reduce contract and still missing
  higher-precision RMSNorm epilogue support.
- `inplace_matmul_all_reduce_add_rms_norm` now runs on the local 2-rank HCCL
  harness in both baseline and PTO paths for a constrained dense seed, with the
  PTO side still depending on a host-side all-reduce contract and currently
  modeling the inplace residual contract through copy-back rather than validated
  native same-buffer aliasing.

## Phase 1 Status

The six seed kernels are now fully classified:

- `grouped_matmul`: baseline and PTO both run
- `apply_rotary_pos_emb`: baseline and PTO both run
- `ffn`: baseline and PTO both run; the PTO rewrite now matches upstream output tiling more closely but remains slower and still relies on a staged pipeline rather than a fused kernel
- `moe_token_permute`: baseline and PTO both run; PTO still depends on a host-precomputed gather map rather than on-device routing primitives
- `flash_attention_score`: baseline and PTO both run; the PTO rewrite now matches upstream tile ownership more closely but still needs masked or online attention generalization, deeper overlap parity, and performance tuning
- `matmul_reduce_scatter`: baseline and PTO both run on the local 2-rank HCCL harness; PTO still depends on a host-orchestrated collective contract rather than on PTODSL MC2 collectives
- `all_gather_matmul`: baseline and PTO both run on the local 2-rank HCCL harness; PTO now follows the upstream local-first wrapped rank-chunk traversal more closely, but it still depends on a host-orchestrated all-gather contract rather than on PTODSL MC2 collectives
- `grouped_mat_mul_all_reduce`: baseline and PTO both run on the local 2-rank HCCL harness for a constrained single-group seed; PTO now follows the upstream splitM/numBlocksN turn loop more closely, but it still depends on a host-orchestrated all-reduce contract rather than on PTODSL MC2 collectives
- `matmul_all_reduce`: baseline and PTO both run on the local 2-rank HCCL harness for a constrained dense seed; PTO now follows the upstream splitM/numBlocksN turn loop more closely, but it still depends on a host-orchestrated all-reduce contract rather than on PTODSL MC2 collectives
- `matmul_all_reduce_add_rms_norm`: baseline and PTO both run on the local 2-rank HCCL harness for a constrained dense seed; PTO now follows the upstream splitM/numBlocksN turn loop plus a staged vector epilogue more closely, and the refined Newton-step PTODSL RMSNorm path is now correctness-green on the validated shapes (`max_abs_diff = 0.00390625`, baseline / PTO `= 65.53%`). The seed still depends on a host-orchestrated all-reduce contract rather than on true PTODSL MC2 collectives.
- `matmul_all_reduce_add_rms_norm` backend note: current `pto-isa` A2/A3 vector `TLOAD/TSTORE` still rejects GM/vector dtype conversion, so backend-legal mixed-precision vector IO is still absent. The current seed no longer depends on that path for correctness, but broader generalized epilogues may still need it later.
- `inplace_matmul_all_reduce_add_rms_norm`: baseline and PTO both run on the local 2-rank HCCL harness for a constrained dense seed; PTO now follows the upstream splitM/numBlocksN turn loop plus a staged vector epilogue more closely, and the refined RMSNorm path is also correctness-green here (`max_abs_diff = 0.00390625`, baseline / PTO `= 54.79%`). The seed still models the inplace residual contract with copy-back because native same-buffer aliasing is not yet validated.

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
