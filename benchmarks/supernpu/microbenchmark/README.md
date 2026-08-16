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

Set `COMPILER_DIR` to a LinxISA v0.58 compiler `bin/` directory and
`LINX_TILEOP_API_ROOT` to a matching TileOP API checkout. Set `LINX_SYSROOT`
to the matching Linx musl sysroot with the C++ runtime overlay, then run:

```bash
bash compile_all.sh all
```

This command is a toolchain validation action, not part of the lightweight
pto-kernels pull-request gate.
