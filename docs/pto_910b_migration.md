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

- baseline median latency: about `0.108 ms`
- PTO median latency: about `0.255 ms`
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
- the local PTODSL matmul-guide swizzle patterns are now available through
  tuning envs, but the current checked A3 default stays on linear traversal
  because it benchmarked better than the `zn2` / `nz2` swizzled variants on
  the current seed shapes

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

### `grouped_matmul_add`

The `grouped_matmul_add` seed now runs end to end in both baseline and PTO
paths for a constrained dense BF16 -> F32 add epilogue slice:

- shape variants:
  - `x_t = [128, 64]`, `x_pto = [64, 128]`, `weight = [128, 128]`,
    `y_init = [64, 128]`
  - `x_t = [128, 128]`, `x_pto = [128, 128]`, `weight = [128, 256]`,
    `y_init = [128, 256]`
- dtype: `bfloat16` inputs, `float32` add/output
- one dense group with `group_list = [K]`
- no routing, quantization, activation, or batched weight list

Current benchmark on this host:

- baseline median latency: about `0.240 ms`
- PTO median latency: about `0.383 ms`
- correctness: both paths pass at `atol=rtol=2e-2` with current
  `max_abs_diff` about `1.14e-5`

The validated PTO rewrite keeps execution on NPU and expresses the seed as two
staged PTO kernels:

1. a tiled BF16 cube matmul stage that accumulates into an F32 temporary GM
   buffer with the same upstream-shaped block traversal used in the current
   grouped-matmul slice
2. a vector add stage that fuses the seeded `y_init` residual into the final

The local PTODSL matmul-guide swizzle traversal is also available on the
matmul stage, but the current checked A3 default stays on linear traversal
because it benchmarked better than the `zn2` variant on this seed.
   F32 output

The active remaining gap is the same cube-family performance gap already shared
by `grouped_matmul`: PTODSL still lacks a reusable source-level preload and
double-buffer pipeline surface matching the full overlap strategy used by the
upstream AscendC grouped GEMM kernels.

### `grouped_matmul_finalize_routing`

The next GMM slice is not blocked on PTO source yet; it is blocked on the
baseline contract exposed by the current host runtime.

What is confirmed on this host:

- the public entrypoint exists as `torch_npu.npu_grouped_matmul_finalize_routing`
- simple dense ND tensors are not enough for a valid call
- the runtime requires routed inputs such as `scale`, `pertokenScale`,
  `logit`, and `rowIndex`
- plain ND `INT8` weights are rejected; the accepted ND path expects routed
  quantized weight dtypes such as `INT32` or supported low-precision formats
- the alternative weight path needs FRACTAL_NZ-style storage metadata rather
  than a plain contiguous 3D PyTorch tensor

Direct minimal probes therefore fail before a stable dense reference slice is
available:

- missing routed tensors are rejected as invalid parameters
- a naive 3D routed probe can crash in the current host runtime before it
  reaches a repeatable benchmarkable state
- the obvious `torch_npu.npu_format_cast` route for probing NZ-format weights
  currently falls into a local TBE-side environment failure on this machine
  (`ModuleNotFoundError: decorator`) before a usable baseline tensor is
  produced

The next implementation step for this kernel is to reproduce the required
quantized routed input contract and weight storage layout in the baseline
harness first. Only then is it worth landing the corresponding PTO slice.

The repository now carries this as a bounded blocked slice rather than leaving
it implicit:

- benchmark spec: [bench/specs/gmm/grouped_matmul_finalize_routing.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/gmm/grouped_matmul_finalize_routing.yaml)
- baseline adapter: [grouped_matmul_finalize_routing.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/gmm/grouped_matmul_finalize_routing.py)
- PTO adapter: [grouped_matmul_finalize_routing.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/gmm/grouped_matmul_finalize_routing.py)

That means the migration harness can now emit a stable blocked report for this
kernel with the measured host-contract failures, and the next engineer does not
need to rediscover them by hand.

### `grouped_matmul_swiglu_quant`

This kernel is now tracked as a bounded blocked slice on this host.

What is confirmed:

- the public entrypoint exists as `torch_npu.npu_grouped_matmul_swiglu_quant`
- the host baseline requires FRACTAL_NZ-formatted int8 weight storage metadata
- plain contiguous PyTorch weights fail with a storage-shape error before a
  stable benchmarkable baseline exists

Tracked slice:

- benchmark spec: [bench/specs/gmm/grouped_matmul_swiglu_quant.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/gmm/grouped_matmul_swiglu_quant.yaml)
- baseline adapter: [grouped_matmul_swiglu_quant.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/gmm/grouped_matmul_swiglu_quant.py)
- PTO adapter: [grouped_matmul_swiglu_quant.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/gmm/grouped_matmul_swiglu_quant.py)

### `grouped_matmul_swiglu_quant_v2`

This kernel is also now tracked as a bounded blocked slice.

What is confirmed:

- the public entrypoint exists as `torch_npu.npu_grouped_matmul_swiglu_quant_v2`
- the host baseline expects list-valued FP8/scale tensors matching the upstream
  ACLNN example contract
- a stable minimal Python baseline slice is not yet reproduced on this host

Tracked slice:

- benchmark spec: [bench/specs/gmm/grouped_matmul_swiglu_quant_v2.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/gmm/grouped_matmul_swiglu_quant_v2.yaml)
- baseline adapter: [grouped_matmul_swiglu_quant_v2.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/gmm/grouped_matmul_swiglu_quant_v2.py)
- PTO adapter: [grouped_matmul_swiglu_quant_v2.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/gmm/grouped_matmul_swiglu_quant_v2.py)

### `swin_attention_ffn`

This kernel is now tracked as a constrained runnable PTO slice.

What is confirmed:

- the validated 910B slice follows the upstream zero-shift formula `y = x1 @ x2 + bias + x3`
- the PTO port uses tiled PTODSL matmul plus a vector add epilogue, with explicit-sync-free source and PTOAS autosync
- the nominal regression shape uses all requested block ids
- this host does not expose `torch_npu.npu_swin_attention_ffn`, so baseline parity is bounded-blocked
- the current checked-in shapes are `smoke=[2,64,128]`, `boundary=[8,64,128]`, `nominal=[48,64,128]` so all PTO variants satisfy the `baseM=128` divisibility contract

Tracked slice:

- benchmark spec: [bench/specs/ffn/swin_attention_ffn.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/ffn/swin_attention_ffn.yaml)
- baseline adapter: [swin_attention_ffn.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/ffn/swin_attention_ffn.py)
- PTO adapter: [swin_attention_ffn.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/ffn/swin_attention_ffn.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/ffn/swin_attention_ffn/kernel.py)

### `swin_transformer_ln_qkv`

This kernel is now tracked as a constrained runnable PTO slice.

What is confirmed:

- the validated 910B slice follows the upstream formula
  `(Q,K,V)=split(layernorm(x) @ weight + bias)`
- the PTO port is PTODSL-only and uses three explicit stages:
  - vector LayerNorm on `[tokens, 128]`
  - tiled cube matmul on `[tokens, 128] x [128, 384]`
  - vector split+bias into `q/k/v`
- the source stays explicit-sync-free and relies on PTOAS autosync
- the nominal regression shape uses all requested block ids
- this host does not expose `torch_npu.npu_swin_transformer_ln_qkv`, so
  baseline parity is bounded-blocked

Current verified PTO shapes on this host:

- smoke: `inputX=[1,64,128]`, `weight=[128,384]`, outputs
  `query/key/value=[1,4,64,32]`
- nominal: `inputX=[8,256,128]`, `weight=[128,384]`, outputs
  `query/key/value=[8,4,256,32]`
- boundary: `inputX=[4,256,128]`, `weight=[128,384]`, outputs
  `query/key/value=[4,4,256,32]`

Current benchmark on this host:

- PTO median latency: about `0.739 ms`
- correctness: passes at `atol=1e-2` with current `max_abs_diff`
  about `4.70e-3`
- block utilization: nominal variant uses all requested block ids with
  `nominal_tokens=2048`, `baseM=128`, `requested_block_dim=24`,
  and `nominal_tiles=48`

Tracked slice:

- benchmark spec: [bench/specs/ffn/swin_transformer_ln_qkv.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/ffn/swin_transformer_ln_qkv.yaml)
- baseline adapter: [swin_transformer_ln_qkv.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/ffn/swin_transformer_ln_qkv.py)
- PTO adapter: [swin_transformer_ln_qkv.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/ffn/swin_transformer_ln_qkv.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/ffn/swin_transformer_ln_qkv/kernel.py)

### `swin_transformer_ln_qkv_quant`

This kernel is now tracked as a bounded blocked Wave 1 slice.

What is confirmed:

- the upstream formula is the quantized Swin path
  `(Q,K,V)=split(dequant(int8_matmul(quant(layernorm(x)), weight, bias)))`
- the public ACLNN documentation for the packaged op marks Atlas A2 / 910B
  unsupported
- this host exposes no Python-visible `torch_npu` or `torch.ops.npu`
  baseline entrypoint for the quantized kernel
- the PTO stack does not yet expose a validated end-to-end
  layernorm -> quantize(int8) -> int8 cube matmul with int32 bias ->
  dequantize -> split path that matches the upstream kernel contract

Tracked bounded shapes:

- smoke: `x=[1,64,128]`, `weight=[128,384]`, outputs
  `query/key/value=[1,4,64,32]`
- nominal: `x=[8,64,128]`, `weight=[128,384]`, outputs
  `query/key/value=[8,4,64,32]`
- boundary: `x=[32,64,128]`, `weight=[128,384]`, outputs
  `query/key/value=[32,4,64,32]`

Current benchmark state on this host:

- baseline: blocked because the host has no Python entrypoint and the
  packaged public docs mark A2 unsupported
- PTO: blocked because the current PTO stack still lacks the full
  quantized Swin LN-QKV surface
- this slice is intentionally in the regression matrix so the failure
  mode stays explicit and stable instead of remaining implicit

Tracked slice:

- benchmark spec: [bench/specs/ffn/swin_transformer_ln_qkv_quant.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/ffn/swin_transformer_ln_qkv_quant.yaml)
- baseline adapter: [swin_transformer_ln_qkv_quant.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/ffn/swin_transformer_ln_qkv_quant.py)
- PTO adapter: [swin_transformer_ln_qkv_quant.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/ffn/swin_transformer_ln_qkv_quant.py)
- PTO kernel placeholder: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/ffn/swin_transformer_ln_qkv_quant/kernel.py)

### `quant_grouped_matmul_inplace_add`

This kernel is now tracked as a bounded blocked slice as well.

What is confirmed:

- the upstream ACLNN/C++ kernel exists in `ops-transformer` for 910B
- this host does not expose a matching `torch_npu` Python entrypoint
- the current PTO workspace therefore cannot yet build a Python-side baseline
  harness for parity

Tracked slice:

- benchmark spec: [bench/specs/gmm/quant_grouped_matmul_inplace_add.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/gmm/quant_grouped_matmul_inplace_add.yaml)
- baseline adapter: [quant_grouped_matmul_inplace_add.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/gmm/quant_grouped_matmul_inplace_add.py)
- PTO adapter: [quant_grouped_matmul_inplace_add.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/gmm/quant_grouped_matmul_inplace_add.py)

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

### `interleave_rope`

The `interleave_rope` seed now runs end-to-end in both baseline and PTO paths
for the first BNSD fp16 slice:

- shape variants: `1 x 1 x 32 x 64` and `2 x 1 x 32 x 64`
- dtype: `float16`
- layout: `BNSD`
- `head_dim = 64`

Current benchmark on this host:

- baseline median latency: about `0.099 ms`
- PTO median latency: about `0.326 ms`
- correctness: both paths pass at `atol=1e-2`, and the current PTO seed lands
  an exact match to the checked-in CPU reference for the validated shapes

The current PTO rewrite keeps execution on NPU but splits the op into two
stages:

1. NPU-side `reshape + transpose` preprocessing to materialize the interleaved
   rope input
2. a PTODSL rotary-half compute kernel over the interleaved tensor

That means the seed is no longer blocked on correctness or basic NPU bring-up,
but it still does not model the upstream on-device even/odd split directly in
PTO source. The active remaining gap is a reusable PTODSL vector-shuffle or
GatherMask-style surface, plus the same upstream vector queue / double-buffer
pipeline gap already shared by the rope family.

### `rope_quant_kvcache`

The `rope_quant_kvcache` seed now runs end-to-end in both baseline and PTO
paths for a constrained 2D token-major slice:

- shape variants:
  - `x = [2, 192]`, `cos = [2, 64]`, `sin = [2, 64]`
  - `x = [4, 192]`, `cos = [4, 64]`, `sin = [4, 64]`
- cache shape: `[B, 8, 1, 64]`
- dtype: `float16`
- `size_splits = [64, 64, 64]`
- single q-head and single k/v-head
- unit quant scales only
- `cache_mode = contiguous`

Current benchmark on this host:

- baseline median latency: about `0.129 ms`
- PTO median latency: about `1.406 ms`
- correctness: both paths pass at `atol=1e-2`; the current PTO slice matches
  the checked-in fp16 half-rope + unit-scale quant/cache reference with
  `max_abs_diff` about `3.52e-3`

The current PTO rewrite keeps execution on NPU and has been moved fully into
staged PTO kernel source for the constrained seed:

1. a PTODSL rotary stage that splits the fused `qkv` input and computes `q/k/v`
2. a PTODSL cache stage that performs explicit `tcvt` plus `tstore` cache writeback for the int8 cache slices

This explicit `tcvt(f16->i8) + tstore(i8)` sequence is intentional. The
current A2/A3 PTO stack does not expose a native fused vec quantized-store path
for this contract: vec `tstore` requires matching source and destination
element types, while `tstore_fp` is aligned with the accumulator quantized
store path rather than a generic vec -> int8 GM store.

The active remaining gap is no longer torch-side implementation. It is a PTO
stack gap around broader cache-update support: the validated seed still assumes
`indices[row] == row`, unit quant scales, and contiguous cache layout. The
general fused split-pack + dynamic cache-index path still needs reusable PTO
surface area.

### `dequant_rope_quant_kvcache`

The `dequant_rope_quant_kvcache` seed now runs end-to-end in both baseline and
PTO paths for a constrained int32 + weight-scale 2D token-major slice:

- shape variants:
  - `x = [2, 192]`, `cos = [2, 64]`, `sin = [2, 64]`, `weight_scale = [192]`
  - `x = [4, 192]`, `cos = [4, 64]`, `sin = [4, 64]`, `weight_scale = [192]`
- cache shape: `[B, 8, 1, 64]`
- dtype: `int32` input, fp16 rope stages, int8 cache output
- `size_splits = [64, 64, 64]`
- single q-head and single k/v-head
- `cache_mode = contiguous`

Current benchmark on this host:

- baseline median latency: about `0.145 ms`
- PTO median latency: about `1.443 ms`
- correctness: both paths pass at `atol=rtol=2e-2`; the current PTO slice
  matches the checked-in dequant + half-rope + quant/cache reference with
  `max_abs_diff` about `1.41e-2`

The current PTO rewrite keeps execution on NPU and moves the constrained fused
slice into staged PTO kernel source:

1. a PTODSL rotary stage that performs `tcvt + mul + tcvt` dequant for the int32 `weight_scale` input and computes `q/k/v`
2. a PTODSL cache stage that performs explicit `tcvt` plus `tstore` cache writeback for `k_cache` and `v_cache`

As with `rope_quant_kvcache`, this uses the legal explicit vec path today:
`tcvt(... -> i8)` followed by same-dtype `tstore`. There is no current fused
vec quantized store contract exposed through PTO-ISA/PTOAS for this seed.

The active remaining gap is the same family-level cache/index generalization as
`rope_quant_kvcache`: the seed still assumes `indices[row] == row`, unit quant
scales, and the constrained contiguous cache contract. The full fused dynamic
cache-index path still needs broader PTO surface area.

### `qkv_rms_norm_rope_cache`

The `qkv_rms_norm_rope_cache` seed is now a real bounded PTO slice for a
constrained 2D fp16 contract:

- shape variants:
  - `qkv = [2, 192]`, `q_gamma = [64]`, `k_gamma = [64]`,
    `cos = [2, 64]`, `sin = [2, 64]`, cache = `[2, 8, 1, 64]`
  - `qkv = [4, 192]`, `q_gamma = [64]`, `k_gamma = [64]`,
    `cos = [4, 64]`, `sin = [4, 64]`, cache = `[4, 8, 1, 64]`
- dtype: `float16`
- `size_splits = [64, 64, 64]`
- single q-head and single k/v-head
- contiguous int8 cache writeback

Current verified state on this host:

- baseline path is blocked because the local `torch_npu` runtime does not
  expose `npu_qkv_rms_norm_rope_cache`
- PTO path is correctness-green on the validated shapes with median about
  `0.7741 ms` and worst `max_abs_diff = 0.005847`
- the current PTO staged slice emits:
  1. a PTODSL rotary stage with row-wise RMSNorm on `q` and `k`
  2. a PTODSL cache stage with explicit `tcvt + tstore` int8 cache writeback

The active remaining gaps are:

- cache-index and cache-update generalization: the validated slice still
  assumes identity row-to-cache lookup and the constrained contiguous cache
  contract
- rope/cache PTODSL surface generalization for broader rotary, interleave, and
  cache-update contracts beyond the current contiguous identity-index seed

### `rope_with_sin_cos_cache`

The `rope_with_sin_cos_cache` seed now has a first PTO-only staged slice for a
constrained ND fp16 half-rope cache-read contract:

- shape variants:
  - `positions = [2]`, `query = [2, 64]`, `key = [2, 64]`,
    `cosSinCache = [8, 128]`
  - `positions = [4]`, `query = [4, 64]`, `key = [4, 64]`,
    `cosSinCache = [8, 128]`
- dtype: `float16`
- head size fixed to `64`
- NeoX half-rope mode only
- current validated slice requires `positions[row] == row`

Current verified state on this host:

- baseline path is blocked because the local `torch_npu` runtime does not
  expose `npu_rope_with_sin_cos_cache`
- PTO path is implemented entirely in PTO source for the constrained slice,
  with cache read and rope compute both staying inside the PTO stack

The active remaining gap is dynamic cache-index generalization: the current PTO
seed consumes contiguous cache rows directly and does not yet cover arbitrary
`positions` lookup semantics across the full operator contract.

### `rotary_position_embedding`

The `rotary_position_embedding` seed now runs end-to-end in both baseline and
PTO paths for a constrained half-mode single-route slice:

- shape variants:
  - `BSND`: `x/cos/sin/out = [2, 32, 1, 128]`
  - `BNSD`: `x/cos/sin/out = [2, 1, 32, 128]`
- dtype: `float16`
- mode: `half`
- head size fixed to `128`

Current benchmark on this host:

- baseline median latency: about `0.100 ms`
- PTO median latency: about `0.178 ms`
- correctness: both paths pass at `atol=rtol=1e-2`; the current PTO slice
  matches the checked-in CPU half-rope reference with `max_abs_diff`
  about `4.31e-3`

The baseline path on this host uses `torch_npu.npu_rotary_mul(..., "half")`,
which matches the supported A2/A3 single-route rotary contract for this
constrained slice even though there is no separate dedicated
`torch_npu.npu_rotary_position_embedding` entrypoint.

The active remaining gap is family-level generalization rather than basic
bring-up: the current PTO slice is validated only for fp16 half-mode with
single-head BSND/BNSD shapes and still does not cover interleave mode, broader
broadcast layouts, or the upstream vector queue / double-buffer pipeline.

### `rotary_position_embedding_grad`

The `rotary_position_embedding_grad` seed now runs end-to-end in the PTO path
for a constrained half-mode single-route backward slice, and the host baseline
path is usable for `dx` parity on this machine:

- shape variants:
  - `BSND`: `dy/x/cos/sin/dx/dcos/dsin = [2, 32, 1, 128]`
  - `BNSD`: `dy/x/cos/sin/dx/dcos/dsin = [2, 1, 32, 128]`
- dtype: `float16`
- mode: `half`
- head size fixed to `128`

Current benchmark on this host:

- baseline median latency: about `0.108 ms`
- PTO median latency: about `0.283 ms`
- correctness:
  - PTO passes for `dx/dcos/dsin` against the checked-in no-reduction backward
    reference at `atol=rtol=2e-2`
  - the baseline `torch_npu.npu_rotary_mul_backward` path is currently
    validated on `dx` only because this host returns zero-filled `dcos/dsin`
    outputs even when `xOptional` is provided

So the current blocker for this backward slice is not PTO codegen but host
baseline completeness. The PTO seed is ready for regression tracking with the
explicit `dx`-only baseline limitation, while broader family work still
includes interleave backward, broadcast-reduction generalization, and the
upstream vector queue / double-buffer pipeline.

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

### `moe_finalize_routing`

The `moe_finalize_routing` slice now runs end-to-end in both baseline and PTO
paths for a constrained combine slice:

- token shapes: `16 x 16`, `256 x 64`, `128 x 128`
- dtype: `float16`
- experts: `4`
- top-1 routing only
- `x2Optional = None`
- `dropPadMode = 0`

Reference contract for this slice:

- `out = x1 + scales * (expandedX[expandedRowIdx] + bias[expandedExpertIdx])`

Current implementation notes on this host:

- the baseline uses `torch_npu.npu_moe_finalize_routing`
- the PTO seed uses a single vector combine kernel with contiguous per-core
  token-row ownership
- the nominal `256 x 64` variant is chosen so the kernel uses all requested
  block ids
- the PTO seed now uses direct scalar routing indices for `expandedRowIdx`
  and `expandedExpertIdx`, with row-at-a-time GM views instead of flattened
  gather-map preloads
- both baseline and PTO sit around `8e-3` max abs diff versus the float32
  reference on the larger validated fp16 variants, so this slice uses
  `atol = 1e-2`
- the remaining MoE-routing gap is reusable on-device routing/index
  materialization rather than basic finalize execution

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_finalize_routing.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_finalize_routing.yaml)
- baseline adapter: [moe_finalize_routing.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_finalize_routing.py)
- PTO adapter: [moe_finalize_routing.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_finalize_routing.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_finalize_routing/kernel.py)

### `moe_finalize_routing_v2`

The `moe_finalize_routing_v2` slice now runs end-to-end in PTO for a
constrained combine path and is tracked with a bounded baseline blocker:

- token shapes: `16 x 16`, `256 x 64`, `128 x 128`
- dtype: `float16`
- experts: `4`
- top-1 routing only
- `x1Optional`, `x2Optional`, `biasOptional`, and `scalesOptional` are all present
- `dropPadMode = 0`

Reference contract for this slice:

- `out = x1 + x2 + scales * (expandedX[expandedRowIdx] + bias[expertIdx])`

Current implementation notes on this host:

- the PTO seed uses a single vector combine kernel with contiguous per-core
  token-row ownership
- the nominal `256 x 64` variant is chosen so the kernel uses all requested
  block ids
- the PTO seed uses direct scalar routing indices for `expandedRowIdx`
  and `expertIdx`, with row-at-a-time GM views instead of flattened gather-map
  preloads
- the current host environment does not expose a Python-visible
  `moe_finalize_routing_v2` baseline entrypoint, so the baseline side remains
  an explicit blocked report instead of a fake parity result

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_finalize_routing_v2.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_finalize_routing_v2.yaml)
- baseline adapter: [moe_finalize_routing_v2.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_finalize_routing_v2.py)
- PTO adapter: [moe_finalize_routing_v2.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_finalize_routing_v2.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_finalize_routing_v2/kernel.py)

### `moe_finalize_routing_v2_grad`

The `moe_finalize_routing_v2_grad` slice now runs end-to-end in PTO for the
constrained top-1 backward contract on this 910B host:

- token shapes: `16 x 16`, `256 x 64`, `128 x 128`
- dtype: `float16`
- experts: `4`
- top-1 routing only
- `scalesOptional` and `biasOptional` are present
- `dropPadMode = 0`
- `activeNum = R * K`

Reference contract for this slice:

- `gradExpandedXOut[expandedRowIdx[i]] = gradY[i] * scales[i]`
- `gradScalesOut[i] = sum((expandedX[expandedRowIdx[i]] + bias[expertIdx[i]]) * gradY[i])`

Current implementation notes on this host:

- the PTO seed is written in PTODSL with a vector `gradExpandedXOut` path and
  an fp32 tile reduction path for `gradScalesOut`
- the nominal `256 x 64` variant is chosen so the kernel uses all requested
  block ids
- the PTO seed uses direct scalar routing indices for `expandedRowIdx`
  and `expertIdx`, with dynamic row scatter for `gradExpandedXOut`
  and a tile reduction path for `gradScalesOut`
- the current host environment does not expose a Python-visible
  `moe_finalize_routing_v2_grad` baseline entrypoint, so the baseline side
  remains an explicit blocked report instead of a fake parity result
- the old PTOAS lowering blocker is closed on the default local toolchain:
  `moe_finalize_routing_v2_grad` now compiles and benchmarks through PTOAS,
  Bisheng, and runtime launch
- PTO correctness is now green on the checked fp16 output contract for all
  three validated shapes

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_finalize_routing_v2_grad.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_finalize_routing_v2_grad.yaml)
- baseline adapter: [moe_finalize_routing_v2_grad.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_finalize_routing_v2_grad.py)
- PTO adapter: [moe_finalize_routing_v2_grad.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_finalize_routing_v2_grad.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_finalize_routing_v2_grad/kernel.py)

### `moe_token_unpermute`

The `moe_token_unpermute` slice now runs end-to-end in both baseline and PTO
paths for a constrained inverse-routing slice:

- token shapes: `8 x 16`, `256 x 64`, `128 x 128`
- dtype: `float16`
- `sortedIndices`: 1D `int32`
- top-1 routing only
- `probsOptional = None`
- `paddedMode = false`
- `restoreShapeOptional = None`

Reference contract for this slice:

- `restored_tokens = permuted_tokens[sorted_indices]`

Current implementation notes on this host:

- the baseline uses `torch_npu.npu_moe_token_unpermute`
- the PTO seed uses a single vector restore kernel with contiguous per-core
  token-row ownership
- the nominal `256 x 64` variant uses all requested block ids
- the current PTO seed still consumes a host-precomputed inverse gather map,
  so the remaining MoE-routing gap is still reusable on-device inverse
  permutation and reorder support rather than basic restore execution

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_token_unpermute.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_token_unpermute.yaml)
- baseline adapter: [moe_token_unpermute.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_token_unpermute.py)
- PTO adapter: [moe_token_unpermute.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_token_unpermute.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_token_unpermute/kernel.py)

### `moe_token_unpermute_grad`

The `moe_token_unpermute_grad` slice targets the constrained top-1 no-probs
backward contract:

- token shapes: `8 x 16`, `256 x 64`, `128 x 128`
- dtype: `float16`
- `probsOptional = None`
- `paddedMode = false`
- `restoreShapeOptional = None`

Reference contract for this slice:

- `permutedTokensGrad[sortedIndices[i]] = unpermutedTokensGrad[i]`
- `probsGradOut ~= 0` for the `probs=None` path on this host

Current implementation notes on this host:

- the baseline uses `torch_npu.npu_moe_token_unpermute_grad`
- on the nominal `256 x 64` variant, this host returns drifted `probs_grad` even
  when `probsOptional = None`; baseline parity is therefore validated on
  `permutedTokensGrad` only, while PTO still validates `probsGradOut`
- the PTO seed uses direct scalar routing indices and row-at-a-time GM stores
- the nominal `256 x 64` variant is chosen so the kernel uses all requested
  block ids
- the remaining MoE-routing gap is reusable on-device inverse-permutation and
  grouped routing primitives rather than basic backward writeback

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_token_unpermute_grad.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_token_unpermute_grad.yaml)
- baseline adapter: [moe_token_unpermute_grad.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_token_unpermute_grad.py)
- PTO adapter: [moe_token_unpermute_grad.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_token_unpermute_grad.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_token_unpermute_grad/kernel.py)

### `moe_re_routing`

The first `moe_re_routing` slice now runs end-to-end in both baseline and PTO
paths on this 910B host:

- token shapes: `8 x 16`, `256 x 64`, `128 x 128`
- count matrices: `2 x 2` and `4 x 4`
- dtype: `float16`
- `per_token_scales` enabled
- `expert_token_num_type = 1`
- `idx_type = 0`

Reference contract for this slice:

- input tokens are laid out in source `rank-major, expert-minor` order
- output tokens are reordered into destination `expert-major, rank-minor` order
- `permute_token_idx` is the gather-index map back into the source token rows
- `expert_token_num` is the per-expert token count

Current implementation notes on this host:

- the baseline uses `torch_npu.npu_moe_re_routing` and is correctness-green on
  all validated shapes
- the PTO seed computes the expert-major remap directly from
  `expert_token_num_per_rank` with scalar prefix scans and row-at-a-time GM
  loads and stores
- the checked rewrite uses compile-time-unrolled `2x2` and `4x4` routing tables
  for the validated slices, which removes the old PTOAS section-local
  loop-carried SCF blocker
- the nominal `256 x 64` variant is chosen so the kernel uses all requested
  block ids
- both baseline and PTO are correctness-green on the validated variants
- the kernel is still scalar-heavy rather than tile-first because the routing
  hot path still uses direct scalar loads/selects

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_re_routing.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_re_routing.yaml)
- baseline adapter: [moe_re_routing.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_re_routing.py)
- PTO adapter: [moe_re_routing.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_re_routing.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_re_routing/kernel.py)

### `moe_token_permute_grad`

The next MoE rewrite slice now lands as a correctness-green runnable kernel on
this 910B host.

What is verified:

- the validated Python-visible host contract is a constrained top-1 no-probs
  path even though the published ACLNN documentation describes a top-k reduce
  formula
- on this machine, `torch_npu.npu_moe_token_permute_grad` behaves as a direct
  row gather from `sorted_indices` and ignores the `indices` tensor when
  `probsOptional=None`
- the PTO kernel mirrors that actual host contract instead of forcing the
  published top-k reduction formula into the regression matrix
- the PTO slice is PTODSL-only, explicit-sync-free, and uses contiguous
  per-core token-row ownership for the gather stage
- the nominal shape uses all requested block ids

Validated shapes:

- smoke: `tokens=[8, 16]`, `gradPermuted=[8, 16]`, `sortedIndices=[8]`
- nominal: `tokens=[256, 64]`, `gradPermuted=[256, 64]`, `sortedIndices=[256]`
- boundary: `tokens=[128, 128]`, `gradPermuted=[128, 128]`,
  `sortedIndices=[128]`

Current measured result:

- baseline median latency: about `0.100 ms`
- PTO median latency: about `0.226 ms`
- `baseline / PTO * 100`: about `44.2%`
- correctness: both baseline and PTO pass with `max_abs_diff = 0.0`

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_token_permute_grad.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_token_permute_grad.yaml)
- baseline adapter: [moe_token_permute_grad.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_token_permute_grad.py)
- PTO adapter: [moe_token_permute_grad.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_token_permute_grad.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_token_permute_grad/kernel.py)

### `moe_token_permute_with_routing_map`

The next MoE routing-map slice now lands as a correctness-green runnable kernel
on this 910B host.

What is verified:

- the validated host entrypoint is
  `torch_npu.npu_moe_token_permute_with_routing_map`
- the current checked-in slice is constrained to top-1 `routing_map`,
  `probsOptional=None`, `dropAndPad=false`, and `numOutTokens=tokens_num`
- both baseline and PTO are correctness-green on all validated shapes
- the PTO kernel is PTODSL-only, explicit-sync-free, and uses the same
  contiguous per-core token-row ownership pattern as the earlier MoE permute
  seed
- the current PTO rewrite keeps the actual reorder on NPU and uses
  host-precomputed gather and inverse-order maps only for the still-missing
  on-device routing-map sort/inverse-permutation primitives

Validated shapes:

- smoke: `tokens=[8, 16]`, `routingMap=[8, 4]`
- nominal: `tokens=[256, 64]`, `routingMap=[256, 8]`
- boundary: `tokens=[128, 128]`, `routingMap=[128, 8]`

Current measured result:

- baseline median latency: about `0.110 ms`
- PTO median latency: about `0.339 ms`
- `baseline / PTO * 100`: about `32.6%`
- correctness: both baseline and PTO pass with `max_abs_diff = 0.0`

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_token_permute_with_routing_map.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_token_permute_with_routing_map.yaml)
- baseline adapter: [moe_token_permute_with_routing_map.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_token_permute_with_routing_map.py)
- PTO adapter: [moe_token_permute_with_routing_map.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_token_permute_with_routing_map.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_token_permute_with_routing_map/kernel.py)

### `moe_token_unpermute_with_routing_map`

The symmetric routing-map unpermute slice is now also correctness-green on
this 910B host.

What is verified:

- the validated host entrypoint is
  `torch_npu.npu_moe_token_unpermute_with_routing_map`
- the current checked-in slice is constrained to top-1 `routing_map`,
  `probsOptional=None`, `dropAndPad=false`, and
  `restoreShape=[tokens_num, hidden_size]`
- both baseline and PTO are correctness-green on all validated shapes
- the PTO kernel is PTODSL-only, explicit-sync-free, and uses the same
  contiguous per-core token-row ownership pattern as the earlier unpermute
  seed
- the current PTO rewrite keeps the actual reorder on NPU and uses a
  host-precomputed gather map only for the still-missing on-device
  routing-map inverse-permutation primitives

Validated shapes:

- smoke: `permutedTokens=[8, 16]`, `routingMap=[8, 4]`
- nominal: `permutedTokens=[256, 64]`, `routingMap=[256, 8]`
- boundary: `permutedTokens=[128, 128]`, `routingMap=[128, 8]`

Current measured result:

- baseline median latency: about `0.104 ms`
- PTO median latency: about `0.226 ms`
- `baseline / PTO * 100`: about `46.3%`
- correctness: both baseline and PTO pass with `max_abs_diff = 0.0`

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_token_unpermute_with_routing_map.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_token_unpermute_with_routing_map.yaml)
- baseline adapter: [moe_token_unpermute_with_routing_map.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_token_unpermute_with_routing_map.py)
- PTO adapter: [moe_token_unpermute_with_routing_map.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_token_unpermute_with_routing_map.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_token_unpermute_with_routing_map/kernel.py)

### `moe_token_unpermute_with_routing_map_grad`

The routing-map backward pair is now also correctness-green on this 910B host.

What is verified:

- the validated host entrypoint is
  `torch_npu.npu_moe_token_unpermute_with_routing_map_grad`
- the current checked-in slice is constrained to top-1 `routing_map`,
  `probsOptional=None`, `dropAndPad=false`, and
  `restoreShape=[tokens_num, hidden_size]`
- the Python-visible no-probs host branch is a row-scatter contract:
  `permutedTokensGradOut[outIndex[i]] = unpermutedTokensGrad[i]`
- both baseline and PTO are correctness-green on all validated shapes
- the PTO kernel is PTODSL-only, explicit-sync-free, and uses direct dynamic
  destination row stores instead of a torch-side shim
- `probs_grad` is unsupported in the no-probs host contract and is therefore
  tracked as a baseline limitation rather than a correctness failure

Validated shapes:

- smoke: `unpermutedTokensGrad=[8, 16]`, `routingMap=[8, 4]`
- nominal: `unpermutedTokensGrad=[256, 64]`, `routingMap=[256, 8]`
- boundary: `unpermutedTokensGrad=[128, 128]`, `routingMap=[128, 8]`

Current measured result:

- baseline median latency: about `0.111 ms`
- PTO median latency: about `0.263 ms`
- `baseline / PTO * 100`: about `42.1%`
- correctness: both baseline and PTO pass with `max_abs_diff = 0.0`

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_token_unpermute_with_routing_map_grad.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_token_unpermute_with_routing_map_grad.yaml)
- baseline adapter: [moe_token_unpermute_with_routing_map_grad.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_token_unpermute_with_routing_map_grad.py)
- PTO adapter: [moe_token_unpermute_with_routing_map_grad.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_token_unpermute_with_routing_map_grad.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_token_unpermute_with_routing_map_grad/kernel.py)

### `moe_token_permute_with_routing_map_grad`

The symmetric routing-map permute backward slice is now also correctness-green
on this 910B host.

What is verified:

- the validated host entrypoint is
  `torch_npu.npu_moe_token_permute_with_routing_map_grad`
- the current checked-in slice is constrained to top-1 `routing_map`,
  `probsOptional=None`, and `dropAndPad=false`
- the Python-visible no-probs host branch is a row-scatter contract:
  `tokenGradOut[sortedIndices[i]] = permutedTokensGrad[i]`
- both baseline and PTO are correctness-green on all validated shapes
- the PTO kernel is PTODSL-only, explicit-sync-free, and uses direct dynamic
  destination row stores instead of a torch-side shim
- `probs_grad` is unsupported in the no-probs host contract and is therefore
  tracked as a baseline limitation rather than a correctness failure

Validated shapes:

- smoke: `permutedTokensGrad=[8, 16]`, `routingMap=[8, 4]`
- nominal: `permutedTokensGrad=[256, 64]`, `routingMap=[256, 8]`
- boundary: `permutedTokensGrad=[128, 128]`, `routingMap=[128, 8]`

Current measured result:

- baseline median latency: about `0.092 ms`
- PTO median latency: about `0.260 ms`
- `baseline / PTO * 100`: about `35.6%`
- correctness: both baseline and PTO pass with `max_abs_diff = 0.0`

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_token_permute_with_routing_map_grad.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_token_permute_with_routing_map_grad.yaml)
- baseline adapter: [moe_token_permute_with_routing_map_grad.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_token_permute_with_routing_map_grad.py)
- PTO adapter: [moe_token_permute_with_routing_map_grad.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_token_permute_with_routing_map_grad.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_token_permute_with_routing_map_grad/kernel.py)

### `moe_gating_top_k`

The first `moe_gating_top_k` slice is now baseline and PTO correctness-green on
this 910B host.

What is verified:

- the validated host entrypoint is `torch_npu.npu_moe_gating_top_k`
- the current checked-in slice is constrained to top-1, 2D input,
  `groupCount=1`, `kGroup=1`, `groupSelectMode=0`, `biasOptional=None`,
  `normType=sigmoid`, `renorm=0`, and `outFlag=false`
- on this Python-visible host contract, `yOut` is `1`, `expertIdxOut` is
  `argmax(x)`, and the third returned tensor stays zeroed `float32`
- both baseline and PTO are correctness-green on all validated shapes
- the PTO kernel is PTODSL-only, explicit-sync-free, and uses a direct top-1
  select stage with no torch-side compute shim
- the validated PTO launch uses `block_dim = 8`

Validated shapes:

- smoke: `x=[8, 16]`
- nominal: `x=[256, 64]`
- boundary: `x=[128, 128]`

Current measured result:

- baseline median latency: about `0.133 ms`
- PTO median latency: about `0.238 ms`
- `baseline / PTO * 100`: about `55.7%`
- correctness: both baseline and PTO pass with `max_abs_diff = 0.0`

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_gating_top_k.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_gating_top_k.yaml)
- baseline adapter: [moe_gating_top_k.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_gating_top_k.py)
- PTO adapter: [moe_gating_top_k.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_gating_top_k.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_gating_top_k/kernel.py)

### `moe_gating_top_k_softmax`

The next Wave-2 gating slice is now baseline and PTO correctness-green on this
910B host.

What is verified:

- the validated host entrypoint is `torch_npu.npu_moe_gating_top_k_softmax`
- the current checked-in slice is constrained to top-1, 2D input,
  and `finishedOptional=None`
- baseline is correctness-green on all validated shapes
- the PTO source is PTODSL-only and explicit-sync-free
- the staged PTO design is:
  1. row-wise softmax in PTO
  2. top-1 select over the routed row in PTO
- PTOAS now accepts the stale PTODSL `pto.make_tensor_view` text spelling, so
  the earlier PTODSL/PTOAS package-format mismatch is no longer blocking this
  kernel
- PTOAS scalar `f16` compare lowering now promotes compare operands before
  EmitC codegen, so the earlier Bisheng `half` compare failure is no longer
  blocking this kernel
- the validated PTO launch uses `block_dim = 8`; the earlier `block_dim = 20`
  path corrupted edge row-chunks on this host for the select stage

Validated shapes:

- smoke: `x=[8, 16]`
- nominal: `x=[256, 64]`
- boundary: `x=[128, 128]`

Current measured result:

- baseline median latency: about `0.137 ms`
- PTO median latency: about `0.396 ms`
- `baseline / PTO * 100`: about `34.7%` on the validated boundary shape and
  about `31.9%` on the nominal shape
- correctness: pass, worst-case `max_abs_diff = 0.00021374225616455078`

Tracked limits:

- top-k fixed to `1`
- `finishedOptional` fixed to `None`
- only the 2D gating contract is validated in this slice
- generalized top-k and finished masking remain open

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_gating_top_k_softmax.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_gating_top_k_softmax.yaml)
- baseline adapter: [moe_gating_top_k_softmax.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_gating_top_k_softmax.py)
- PTO adapter: [moe_gating_top_k_softmax.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_gating_top_k_softmax.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_gating_top_k_softmax/kernel.py)

### `moe_gating_top_k_softmax_v2`

The next gating slice is now tracked as a real PTO regression kernel on this
910B host, with a bounded baseline blocker instead of an untracked gap.

What is verified:

- the current checked-in slice is constrained to top-1, 2D input, `renorm=0`,
  `finishedOptional=None`, and `outputSoftmaxResultFlag=false`
- the PTO source is PTODSL-only and explicit-sync-free
- the staged PTO design is:
  1. row-wise softmax in PTO
  2. top-1 select over the softmax row in PTO
- the validated PTO launch uses `block_dim = 8`
- this host does not expose a Python-visible baseline entrypoint for
  `moe_gating_top_k_softmax_v2` in either `torch_npu` or `torch.ops.npu`, so
  baseline parity is bounded-blocked on this machine
- the PTO slice is now correctness-green with the rebuilt local PTOAS
  `build-19` toolchain, which lowers the stage-2 unrolled `f16` select path
  through EmitC and Bisheng successfully
- `scripts/source_env.sh` now prefers the local PTOAS build tree over the
  stale packaged `ptoas` binary on `PATH`, so the validated workspace flow uses
  the same working compiler path by default

Validated shapes:

- smoke: `x=[8, 16]`
- nominal: `x=[256, 64]`
- boundary: `x=[128, 128]`

Current measured result:

- baseline: blocked on missing Python-visible runtime entrypoint
- PTO median latency: about `0.362 ms`
- correctness: pass, worst-case `max_abs_diff = 0.00021374225616455078`
- nominal block utilization: `requested_block_dim = 8`, `uses_all_blocks = true`
- latest bounded report:
  [report.json](/home/zhouruoyu/github/pto-kernels/bench/generated/moe/moe_gating_top_k_softmax_v2/report.json)

Tracked limits:

- top-k fixed to `1`
- `renorm` fixed to `0`
- `finishedOptional` fixed to `None`
- `outputSoftmaxResultFlag` fixed to `false`
- only the 2D gating contract is validated in this slice
- `renorm=1`, optional softmax output, and 3D inputs remain open
- the remaining live blocker on this host is baseline availability, not PTO
  lowering

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_gating_top_k_softmax_v2.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_gating_top_k_softmax_v2.yaml)
- baseline adapter: [moe_gating_top_k_softmax_v2.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_gating_top_k_softmax_v2.py)
- PTO adapter: [moe_gating_top_k_softmax_v2.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_gating_top_k_softmax_v2.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_gating_top_k_softmax_v2/kernel.py)

### `moe_compute_expert_tokens`

The next Wave-2 routing/count slice now has a fully verified baseline contract
on this 910B host, and the direct PTODSL port now compiles and runs through the
full PTO stack.

What is verified:

- the host entrypoint is `torch.ops.npu.npu_moe_compute_expert_tokens`
- the validated contract is `out[e] = upper_bound(sortedExperts, e)`, i.e. the
  exclusive end position for each expert in the sorted row list
- both baseline and PTO are correctness-green on all validated shapes
- the PTO source is PTODSL-only, explicit-sync-free, and uses contiguous
  per-core expert-range ownership
- the PTO rewrite now keeps the running upper-bound position as an SSA
  loop-carried scalar and stores it once per expert, which fixes the earlier
  multiblock GM scalar corruption on the nominal and boundary shapes

Validated shapes:

- smoke: `sortedExperts=[64]`, `numExperts=8`
- nominal: `sortedExperts=[4096]`, `numExperts=64`
- boundary: `sortedExperts=[8192]`, `numExperts=128`

Current measured result:

- baseline median latency: about `0.102 ms`
- PTO median latency: about `0.776 ms`
- `baseline / PTO * 100`: about `13.2%`
- correctness: both paths pass with `max_abs_diff = 0`

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_compute_expert_tokens.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_compute_expert_tokens.yaml)
- baseline adapter: [moe_compute_expert_tokens.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_compute_expert_tokens.py)
- PTO adapter: [moe_compute_expert_tokens.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_compute_expert_tokens.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_compute_expert_tokens/kernel.py)
  backend path rather than frontend SCF emission

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_compute_expert_tokens.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_compute_expert_tokens.yaml)
- baseline adapter: [moe_compute_expert_tokens.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_compute_expert_tokens.py)
- PTO adapter: [moe_compute_expert_tokens.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_compute_expert_tokens.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_compute_expert_tokens/kernel.py)

### `moe_init_routing`

The next Wave-2 routing slice now runs end-to-end in both baseline and PTO
paths on the real `moe_init_routing` host entrypoint, with one explicit
constrained contract to keep the first port faithful and runnable.

What is verified:

- the host entrypoint is `torch.ops.npu.npu_moe_init_routing`
- the validated slice uses `x` as 2D `fp16`, `row_idx` and `expert_idx` as 2D
  `int32` tensors with shape `[tokens, 1]`
- the PTO source is PTODSL-only, explicit-sync-free, and uses the same
  contiguous per-core row ownership pattern as the current MoE gather kernels
- no torch-side compute shim exists in the PTO path; the PTO kernel consumes
  the real `row_idx` / `expert_idx` contract and only uses a flattened gather
  map derived from `row_idx` for the constrained gather stage

Validated shapes:

- smoke: `x=[16,16]`, `row_idx/expert_idx=[16,1]`
- nominal: `x=[256,64]`, `row_idx/expert_idx=[256,1]`
- boundary: `x=[128,128]`, `row_idx/expert_idx=[128,1]`

Current slice limits:

- top-1 only
- `expert_idx` is already grouped by expert on input
- on-device sort is not implemented yet

Tracked slice:

- benchmark spec: [bench/specs/moe/moe_init_routing.yaml](/home/zhouruoyu/github/pto-kernels/bench/specs/moe/moe_init_routing.yaml)
- baseline adapter: [moe_init_routing.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ops_transformer/moe/moe_init_routing.py)
- PTO adapter: [moe_init_routing.py](/home/zhouruoyu/github/pto-kernels/bench/adapters/ptodsl/moe/moe_init_routing.py)
- PTO kernel: [kernel.py](/home/zhouruoyu/github/pto-kernels/python/pto_kernels/ops/moe/moe_init_routing/kernel.py)

### `moe_init_routing_v2`

`moe_init_routing_v2` is now a real Wave-2 prototype on this 910B host for the
dropless top-1 grouped-expert slice. Both paths use the real
`torch.ops.npu.npu_moe_init_routing_v2` contract with:

- `topK = 1`
- `dropPadMode = 0`
- `expertTokensNumType = 1`
- `expertTokensNumFlag = true`
- pre-grouped `expert_idx`

The PTO seed covers:

- `expandedXOut`
- `expandedRowIdxOut`
- `expertTokensCountOrCumsumOut`
- `expertTokensBeforeCapacityOut`

The remaining generalization blocker is still on-device routing sort. This
first slice keeps `expert_idx` pre-grouped so the PTODSL kernel can mirror the
host contract without a torch-side compute shim.

Current bounded blockers on this host:

- baseline: `torch.ops.npu.npu_moe_init_routing_v2` segfaults for the validated
  dropless top-1 grouped-expert probe
- PTO: the checked slice is now correctness-green on all three validated
  variants. The remaining PTO limitation is not legality; the current
  count/cumsum path is still scalar-heavy and the slice still relies on
  pre-grouped `expert_idx` instead of on-device routing sort.

### `flash_attention_score`

The shared dense-attention PTO path is back to correctness-green on this A3
host after keeping the row-wise softmax on the last validated implementation
shape. The attempted fused row-expand rewrite was not kept in the checked-in
kernel because correctness takes priority here.

Validated shapes:

- smoke:
  `query/key/value/out=[1,1,32,64]`, `scores=[32,32]`
- nominal:
  `query/key/value/out=[1,1,64,64]`, `scores=[64,64]`
- boundary:
  `query/key/value/out=[1,1,32,128]`, `scores=[32,32]`

Current benchmark on this host:

- baseline median latency: about `0.153 ms`
- PTO median latency: about `0.414 ms`
- `baseline / PTO * 100`: about `36.9%`
- correctness: both paths pass, current PTO `max_abs_diff` about `1.25e-3`

The PTO seed remains the staged dense path from the shared helper:
tiled `QK`, row-wise softmax, and tiled `PV`. The current remaining work is no
longer basic legality on this kernel. The remaining work is overlap and later
attention generalization, not basic correctness on the constrained dense slice.

### `attention_update`

The `attention_update` slice now runs end-to-end in both baseline and PTO paths
for a constrained Wave-3 attention-update contract:

- shape variants:
  - `lse0/lse1 = [8]`, `local_out0/local_out1/out = [8, 16]`
  - `lse0/lse1 = [256]`, `local_out0/local_out1/out = [256, 64]`
  - `lse0/lse1 = [128]`, `local_out0/local_out1/out = [128, 128]`
- dtype: `float32` for `lse`, `float16` for `localOut/out`
- `sp = 2`
- `updateType = 0`

Reference contract for this slice:

- `out = out0 * softmax([lse0, lse1])[0] + out1 * softmax([lse0, lse1])[1]`
- `lseOut` is omitted in this constrained seed

Current benchmark on this host:

- baseline median latency: about `0.091 ms`
- PTO median latency: about `0.237 ms`
- `baseline / PTO * 100`: about `38.3%`
- correctness: both paths pass at `atol=rtol=2e-3`, with current PTO
  `max_abs_diff` about `9.77e-4`

The current PTO seed is a single vector kernel that mirrors the row-wise
log-sum-exp merge structure of the upstream operator for the validated `sp=2`
slice and now keeps the hot path tile-first by carrying the `lse` scalars in
backend-legal `1xD` fp32 row tiles with `valid_col=1`, instead of scalar loads
and scalar select in the inner loop. The remaining gap is not basic execution
anymore; it is frontend and lowering generalization for list-valued SP inputs
and the optional `lseOut`
branch needed by the full operator contract.

### `ring_attention_update`

The first `ring_attention_update` PTO slice now runs end to end through PTODSL
and PTOAS for a constrained `TND` contract with `N=1`, `sp=2`, fp16 attention
outputs, and fp32 softmax max/sum tensors in the validated repeated-last-dim-8
form. The PTO kernel follows the upstream row-wise merge formula:

- `softmax_max = max(prev_softmax_max, cur_softmax_max)`
- `softmax_sum = prev_sum * exp(prev_max - softmax_max) + cur_sum * exp(cur_max - softmax_max)`
- `attn_out = prev_attn_out * prev_factor / softmax_sum + cur_attn_out * cur_factor / softmax_sum`

Validated shapes:

- smoke:
  `prevAttnOut/curAttnOut/attnOut=[8,1,16]`,
  `prevSoftmaxMax/prevSoftmaxSum/curSoftmaxMax/curSoftmaxSum/softmaxMaxOut/softmaxSumOut=[8,1,8]`
- nominal:
  `prevAttnOut/curAttnOut/attnOut=[256,1,64]`,
  `prevSoftmaxMax/prevSoftmaxSum/curSoftmaxMax/curSoftmaxSum/softmaxMaxOut/softmaxSumOut=[256,1,8]`
- boundary:
  `prevAttnOut/curAttnOut/attnOut=[128,1,128]`,
  `prevSoftmaxMax/prevSoftmaxSum/curSoftmaxMax/curSoftmaxSum/softmaxMaxOut/softmaxSumOut=[128,1,8]`

Current verified state on this host:

- baseline: blocked, because the runtime does not expose a Python-visible
  `ring_attention_update` entrypoint
- PTO: correctness-green on the constrained slice
- PTO median latency: about `0.281 ms`
- correctness: pass, current worst `max_abs_diff` about `1.22e-4`

The current PTO seed is also tile-first now: the repeated-last-dim-8 softmax
metadata path is carried through `1x8` tile `row_max` and `row_sum`, then
broadcast back into the attention rows with `row_expand`, instead of using
scalar loads or scalar select inside the hot path. The remaining gaps are
surface-level rather than core math legality: list-valued SP inputs, broader
layout support, and a host-visible baseline entrypoint for parity.

### `fused_infer_attention_score`

The first `fused_infer_attention_score` slice now has a verified green baseline
on this host for a constrained cache-backed inference contract:

- `batch=1`
- `q_heads=kv_heads=1`
- `total_blocks=1`
- `block_table=[[0]]`
- no quant, no mask, no rope, no shared prefix
- `input_layout="BNSD"`
- cache tensors use `[blocknum, blocksize, H]`

Validated shapes:

- smoke:
  `query=[1,1,16,16]`, `k_cache/v_cache=[1,16,16]`
- nominal:
  `query=[1,1,64,64]`, `k_cache/v_cache=[1,64,64]`
- boundary:
  `query=[1,1,32,128]`, `k_cache/v_cache=[1,128,128]`

Latest reevaluation on the rebuilt local PTOAS toolchain shows the shared
backend legality blocker is fixed: the current constrained PTO slice now
compiles and benchmarks through Bisheng on this host after moving off the
illegal `8x16` cube tiles and onto legal `16x*` cube tiling.

The remaining parity issue on the nominal `64x64` infer variant turned out to
be launch-policy-specific on A3 rather than a math or lowering bug. Keeping the
softmax stage at `block_dim=1` for `q_seq <= 64` restores correctness across
all three checked variants; the larger multiblock softmax setting still
compiles, but drifts numerically on this host.

Current benchmark on this host:

- baseline median latency: about `0.177 ms`
- PTO median latency: about `0.484 ms`
- `baseline / PTO * 100`: about `36.6%`
- baseline correctness: pass
- PTO correctness: pass, `max_abs_diff` about `5.49e-4`

The remaining gap for this slice is performance, not correctness or backend
legality. The checked-in PTO path now uses:

- legal `16x*` cube tiling for `QK` and `PV`
- fp32 row-softmax in the staged infer path
- a smoke-shape `softmax_block_dim=1` launch to avoid over-partitioning the
  `16x16` case, while nominal and boundary variants still use all requested
  block ids

The constrained dense attention seed now runs end-to-end in both baseline and
PTO paths:

- shape: `B=1`, `N=1`, `S=32`, `D=64`
- layout: `BNSD`

### `prompt_flash_attention`

The first `prompt_flash_attention` PTO slice now runs end to end in both
baseline and PTO paths on this host for a constrained dense prompt-attention
contract:

- `batch=1`
- `q_heads=kv_heads=1`
- `input_layout="BNSD"`
- no mask, no quant, no page attention, no shared prefix
- `actual_seq_lengths` and `actual_seq_lengths_kv` fixed to full lengths

Validated shapes:

- smoke:
  `query/key/value/out=[1,1,16,16]`
- nominal:
  `query/out=[1,1,64,64]`, `key/value=[1,1,128,64]`
- boundary:
  `query/out=[1,1,32,128]`, `key/value=[1,1,128,128]`

Current benchmark on this host:

- baseline median latency: about `0.419 ms`
- PTO median latency: about `0.508 ms`
- `baseline / PTO * 100`: about `82.4%`
- baseline correctness: pass
- PTO correctness: pass, `max_abs_diff` about `6.54e-4`

The checked-in PTO path reuses the staged dense-attention pipeline already
validated for `flash_attention_score` and `fused_infer_attention_score`:

- tiled `QK` ownership with legal `16x*` cube tiles
- fp32 staged row-softmax
- tiled `PV` ownership
- smoke-shape `softmax_block_dim=1` to avoid over-partitioning the `16x16`
  case, while nominal and boundary variants still use the regular multiblock
  launch

The active gap for this slice is performance tuning and later prompt-attention
semantic expansion, not correctness or baseline visibility.

### `incre_flash_attention`

The first `incre_flash_attention` PTO slice now runs end to end in both
baseline and PTO paths on this host for a constrained decode contract:

- `q_seq=1`
- `batch=1`
- `q_heads=16`
- `kv_heads=1`
- `input_layout="BNSD"`
- no mask, no quant, no block table, no kv padding
- `actual_seq_lengths` fixed to the full kv sequence length

Validated shapes:

- smoke:
  `query/out=[1,16,1,16]`, `key/value=[1,1,16,16]`
- nominal:
  `query/out=[1,16,1,64]`, `key/value=[1,1,128,64]`
- boundary:
  `query/out=[1,16,1,128]`, `key/value=[1,1,128,128]`

Current benchmark on this host:

- baseline median latency: about `0.135 ms`
- PTO median latency: about `0.562 ms`
- `baseline / PTO * 100`: about `24.1%`
- baseline correctness: pass
- PTO correctness: pass, `max_abs_diff` about `5.57e-4`

The checked-in PTO path reuses the staged dense-attention pipeline already
validated for `prompt_flash_attention` and `fused_infer_attention_score`,
while matching the decode branch by keeping `q_seq=1` in the public contract
and flattening the `16` query heads into the staged dense row dimension:

- legal `16x*` cube tiling for `QK` and `PV`
- fp32 staged row-softmax
- no mask, quant, or page-attention features

The active remaining gap for this slice is performance tuning and later decode
semantic expansion rather than correctness or host baseline visibility.

### `recurrent_gated_delta_rule`

The first `recurrent_gated_delta_rule` slice is now tracked as a bounded Wave 3
prototype on this A3 host.

Validated constrained contract:

- `batch=1`
- `nv=nk=16`
- `q/k/v/state/beta` use `bfloat16`
- `g` uses `float32`
- `gk=None`
- `actual_seq_lengths` fixed to the full sequence length
- `ssm_state_indices=arange(T)`
- `num_accepted_tokens=ones`

Validated shapes:

- smoke:
  `query/key=[2,16,16]`, `value=[2,16,16]`, `state=[2,16,16,16]`
- nominal:
  `query/key=[4,16,64]`, `value=[4,16,64]`, `state=[4,16,64,64]`
- boundary:
  `query/key=[8,16,128]`, `value=[8,16,128]`, `state=[8,16,128,128]`

Current benchmark on this host:

- baseline median latency: about `0.148 ms`
- baseline correctness: pass on `out` only at `atol=rtol=0.15`
- worst validated `out_max_abs_diff`: about `0.1335`
- PTO: blocked

The baseline uses `torch_npu.npu_recurrent_gated_delta_rule_functional` so both
`out` and `final_state` are observable, but the current bounded slice only
validates `out` against the CPU reference. `final_state` remains a host-contract
gap for this first slice.

The PTO stop is now narrower than the earlier frontend-surface blocker.
PTODSL and PTOAS already expose the reusable stack pieces needed by the
checked slice, including `pto.gemv(...)` and the column-broadcast binops
used by the rank-1 state-update composition. The current checked kernel has
been rewritten into row-specialized tile-first state-update stages to avoid
the earlier dynamic-`textract` compile blocker, and those stages now build
through PTOAS/Bisheng on this A3 host.

The live blocker is runtime execution of the generated kernel itself:

- the prebuilt stage artifacts exist under `/tmp/recurrent_row_specialized`
- `stage_proj`, `stage_out`, `stage_scale`, and all `16`
  `stage_state_update_row_*` kernels build successfully
- bypassing PTODSL JIT rebuild and loading the prebuilt `kernel.so`
  artifacts directly through
  [scripts/repro_recurrent_stage_runtime.py](/home/zhouruoyu/github/pto-kernels/scripts/repro_recurrent_stage_runtime.py)
  proves `stage_proj` executes in about `2.27 ms` on the smoke slice and writes
  [repro_proj.json](/tmp/recurrent_row_specialized/repro_proj.json)
- the first row-specialized state-update kernel,
  `stage_state_update_row_000`, still hangs at runtime on A3
- that hang reproduces even with `block_dim = 1`, so it is not just a
  multiblock launch issue
- the direct bounded repro is now:
  `timeout 20s python3 scripts/repro_recurrent_stage_runtime.py --stage state_update_row_000 --block-dim 1`
- the bisect helper
  [scripts/bisect_recurrent_state_update_runtime.py](/home/zhouruoyu/github/pto-kernels/scripts/bisect_recurrent_state_update_runtime.py)
  now rebuilds patched `stage_state_update_row_000` variants with the same
  `bisheng` path used by PTODSL
- patched `no_exp` and `no_extract_term` variants still time out, so the live
  fault is not explained by `TEXP` or the late `TEXTRACT` addend path alone
- `minimal_state_store` (`state load -> state store`) and
  `minimal_state_roundtrip` (`state load -> f32 -> bf16 -> store`) also still
  time out, so the live fault survives even after removing the recurrent
  update arithmetic entirely
- the hand-written backend repro
  [scripts/repro_recurrent_manual_state_store.py](/home/zhouruoyu/github/pto-kernels/scripts/repro_recurrent_manual_state_store.py)
  now splits the runtime path directly:
  `single_row_store_only` succeeds for both `bf16` and `fp16`, while
  both `single_row_load_only` and `single_row_roundtrip`
  (`load -> tcvt -> tcvt -> store`) still time out
- that timeout reproduces even when the source is a fresh standalone source
  buffer rather than the recurrent state tensor, so the issue is not specific
  to the recurrent wrapper inputs or to `bf16` alone
- a direct `copy_state` store-only variant no longer times out; it fails with
  an A3 AI Core illegal-instruction fault (`507015`, reported as unaligned UUB
  / illegal instruction in the vector kernel), which is a stronger backend
  signal than the old generic timeout

So this kernel is no longer blocked by missing PTODSL surface area. It is
currently blocked by the A3 backend/runtime behavior of the minimal vector GM
load or load-consumer pipeline for the emitted `1x16` VEC row shape used by
the row-specialized state-update stage.
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
- `flash_attention_score`: baseline and PTO both run again under the rebuilt local `ptoas`; the shared dense-attention cube-section legality regression is fixed, and the remaining work is deeper overlap parity plus masked or online attention generalization
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
