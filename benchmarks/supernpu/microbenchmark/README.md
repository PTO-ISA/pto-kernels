# SuperNPU microbenchmarks

The active corpus is organized by the v0.58 execution model:

| Directory | Contract |
| --- | --- |
| `scalar/` | scalar/GPR operations |
| `vector/` | VEC elementwise and SFU complex tile operations |
| `memory/` | TLSU memory and tile movement |
| `cube/` | CUBE matrix operations |

Deleted operations are not generated or compiled. Imported cases that used
deleted names are retained only under `../status/legacy/deleted-operation-cases/`.

The CUBE corpus uses PTO ISA 0.58.3 persistent CELL operands:
`CubeTileM32` for A, `CubeTileN8` for B, `CubeAccumulatorM32` for D/C, and
explicit `TLOAD_CUBE` / `TSTORE_CUBE` conversion boundaries. Generated Local
CUBE cases keep M at or below 32. `TMATMULMX` is not generated until the
workload can supply the exact optional E8M0 scale schema and an independent
result oracle; TGEMV awaits the same workload-level M=1 oracle.

CUBE accumulator element types are independent of the input type: floating
inputs accumulate as FP32, signed integer inputs as S32, and unsigned integer
inputs as U32. Bias GM/local tiles use the same accumulator type. Therefore an
FP16 `TMATMUL.BIAS` fixture loads an FP32 1xN Bias and may still select
F32-to-F16 output conversion for D.

Set `COMPILER_DIR` to a LinxISA v0.58 compiler `bin/` directory and
`LINX_TILEOP_API_ROOT` to a matching TileOP API checkout. Set `LINX_SYSROOT`
to the matching Linx musl sysroot with the C++ runtime overlay, then run:

```bash
bash compile_all.sh all
```

This command is a toolchain validation action, not part of the lightweight
pto-kernels pull-request gate.
