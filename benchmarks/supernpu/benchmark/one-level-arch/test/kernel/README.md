# Test Kernel — Operator Test Suites

Per-operator test code and build scripts. Each operator directory has a
`Makefile`, a `compile.all` (typical configs), and `src/`. All suites include
`../../common/Makefile.common`.

## Directory Structure

```
test/kernel/
├── broadcast/        element_wise/    matmul/
├── concat/           reduction/{reducemax_col,row,...}
├── control/          gather/          sort/
└── transpose/
```

## Operator Test Status

| Operator | Configs | Status | Notes |
|----------|---------|--------|-------|
| matmul | 1 | △ | PTO ISA 0.58.3 Local CELL compile smoke; runtime evidence pending |
| transpose | 8 | ✓ | 3D~6D; __half, int32_t |
| reduction | 8 | ✓ | row/col max/sum; int32_t, __half |
| gelu | 8 | ✓ | exact (erf) and tanh; __bf16, __half |
| broadcast | 5 | ✓ | 2D~5D; vectorized |
| gather | 4 | ✓ | large-scale, power-of-2 |
| concat | 4 | ✓ | gather/scatter |
| control | 1 | △ | pure tile-op; run gfsim with `-s core.singleTierMode=true`; `.data` via `gen_data.py` |
| sort | 1 | △ | topk |

(Configs reflect `compile.all` typical scenarios; `△` = compiles but needs special run flags / generated data.)

## Build

### Single config

```bash
cd test/kernel/matmul

# FP16 Local CELL matmul with FP32 accumulator/output
make TESTCASE=matmul TYPE=GMMA MODE=MASK_FP16 \
    M=256 N=256 K=256 tM=16 tN=16 tK=64
```

### Batch / full

```bash
cd test/kernel/matmul && bash compile.all        # per-operator
./compile_all.sh one-level                        # whole backend (from SuperNPU root)
```

### Makefile parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `TESTCASE` | Test case name | `matmul` |
| `TYPE` | Operator type (matmul) | `GMMA` (CELL implementation) |
| `MODE` | Input mode | `MASK_FP16`, `MASK_FP32` |
| `M/N/K` | Matrix dimensions | `M=256 N=2048 K=2048` |
| `tM/tN/tK` | Tile sizes | `tM=16 tN=16 tK=64` (`tM <= 32`) |
| `COMPILER_DIR` | Compiler path | `/path/to/linx/bin` |
| `PLAT` | Platform | `linx` (default), `cpu` |

### Build targets

```bash
make TESTCASE=<case> all      # compile
make TESTCASE=<case> diss     # disassembly
make TESTCASE=<case> sim      # run in simulator
make TESTCASE=<case> debug    # debug mode
make clean                    # clean current operator
make clean_all                # clean all
```

## Operator Details

### Matmul
- `src/matmul_gmma.cpp` — active Local CUBE_M16/M32 × CUBE_N8 smoke with
  explicit FP32 accumulator/output and CUBE GM conversion boundaries.
- Pre-CELL MASK/HIF4/A16W4 drivers are archived under
  `status/legacy/one-level-cube-v058-incomplete/`.

### Flash Attention
- The old one-level FA programs are archived until rebuilt with exact CELL,
  Shared publication, postprocess, and independent runtime-oracle coverage.

### Transpose (8 configs) — see [`transpose/README.md`](transpose/README.md)
2D / 4D(A,B) / 6D(≡5D); __half, int32_t.

### Reduction (8 configs) — see [`reduction/README.md`](reduction/README.md)
`reducemax_{col,row}/`, `reducesum_{col,row}/`; int32_t, __half.

### GELU (8 configs)
`src/gelu.cpp`; __bf16/__half; shapes 24_8_1024 / 128_1024; exact & tanh.

### Broadcast (5 configs) — 2D/3D/4D×2/5D.

### Gather (4 configs) — large (200000,875000), medium (754), power-of-2 (131072).

### Concat (4 configs) — `concat_gather` (2) + `concat_scatter` (2); int32_t, __half.

### Control — `control/hashtable_lookup_simd` (pure tile-op; single-tier gfsim).

### Sort — `sort/topk`.

## Known Issues
- Archived one-level CUBE/FA programs are not active PTO ISA 0.58.3 evidence.
- `control` needs `-s core.singleTierMode=true` on gfsim; `.data` from `gen_data.py`.

## Adding a Test
1. Create `test/kernel/<operator>/` with `src/`, `Makefile`, `compile.all`.
2. Mirror in the other backend.
3. `include ../../common/Makefile.common` (adjust depth if nested).

## See Also
- [Top-level README](../../README.md)
- [Operator implementations](../kernels/README.md)
