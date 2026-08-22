# Kernels — Operator Implementations

Header-only operator implementations, organized by function. All operators use
the PTO tile-programming paradigm via `<common/pto_tileop.hpp>` and C++ templates
for type/dimension parameterization.

## Operator List

> **DeepSeek 迁移算子**：`deepseek/` 子目录保留当前非-CUBE active
> tile 算子；旧 multilayer-recompute GEMM 已移入 legacy，不计入当前验证。
> 详见 [`deepseek/TileKernels迁移说明.md`](deepseek/TileKernels迁移说明.md) 与各模块 README。

### 1. Matmul
- The active executable smoke is the CELL-based
  `test/kernel/matmul/src/matmul_gmma.cpp`.
- Pre-CELL generic/MX headers are archived under
  `status/legacy/one-level-cube-v058-incomplete/`.

### 2. Flash Attention
- The imported one-level FA headers mixed pre-CELL matrix storage and deleted
  CUBE APIs without a current oracle. They are non-normative legacy material.

### 3. Broadcast — `broadcast/`
- `broadcast.hpp` — base; `broadcast_07/019/039/Hunyuan.hpp` — 2D~5D shapes;
  `broadcast_vec_*.hpp` — vectorized; `broadcast_mscatter/nocopyout/nomg/simple.hpp` — variants.

### 4. Reduction — `reduction/` (see [`reduction/README.md`](reduction/README.md))
- `reducemax_{colvec,rowvec}.hpp`, `reducesum_{colvec,rowvec}.hpp` — base max/sum.
- `*_single_tree.hpp` — multi-stage tree reduction.
- `*_unalign_120_8.hpp` — 3D unaligned (120×8).
- `cumsum_{colvec,rowvec}.hpp`, `reduceprod_{colvec,rowvec}.hpp`.

### 5. GELU — `element_wise/`
- `gelu.hpp` — polynomial-fitting; exact (erf) and tanh approximation.
- `gelu_origin.hpp` — original erf/tanh implementation.

### 6. Gather — `gather/`
- `gather.hpp` — large-scale, various indexing modes.

### 7. Concat — `concat/`
- `concat_gather.hpp` — gather-based; `concat_scatter.hpp` — scatter-based.

### 8. Transpose — `transpose/`
- `transpose.hpp` — 3D~6D; `transpose_vector_007/050.hpp` — vectorized.

### 9. Control — `control/`
- `hashtable_lookup_simd.hpp` — pure tile-op kernel (no SIMT). Runs on gfsim
  with `-s core.singleTierMode=true`.

### 10. Sort — `sort/` (see [`sort/README.md`](sort/README.md))
- `topk.hpp` / `topk_pto.hpp` — Top-K via radix-bucket histogram.

### 11. DeepSeek 迁移算子 — `deepseek/` (see [`deepseek/README.md`](deepseek/README.md))
- `_compile_test.cpp` 实例化当前 active 非-CUBE kernel。
- multilayer-recompute 位于 `status/legacy/one-level-cube-v058-incomplete/`。

### Utils — `utils/`
- `layout_transform.hpp` — ND→ZZ / ND→NN offset calculation.

## Design Principles
1. **Header-only** — easy integration/reuse.
2. **PTO paradigm** — unified tile-operation interface.
3. **Templated** — type and dimension parameterization.
4. **Optimization-oriented** — multiple variants per scenario.

## Usage

Use the separately versioned Linx-TileOP-API CUBE_M16/M32 and CUBE_N8 types;
do not include archived one-level CUBE headers.

## Optimization Tips
- Pick `tM/tN/tK` per hardware.
- Choose dtype per precision.
- Use `reuseA/reuseB` for repeated computation.
- Prefer vectorized variants (`vec`/`vector`) where available.

## See Also
- [Top-level README](../../README.md)
- [Test suites](../test/kernel/README.md)
