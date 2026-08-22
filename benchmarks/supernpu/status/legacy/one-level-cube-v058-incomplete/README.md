# Retired incomplete one-level CUBE examples

`fa_2d_unroll_gmma.cpp` is preserved here as non-normative history. It was an
SPMD programming-model sketch, not a validated PTO ISA 0.58 workload:

- it used pre-CELL `TileLeft`/`TileRight` matrix storage;
- it used ordinary `TLOAD`/`TSTORE` at CUBE conversion boundaries;
- it emitted deleted `TMATMUL_FIXP` calls and ordinary Vec destinations;
- its collective Shared-matrix assumptions had no executable result oracle.

The active Makefile and compile list were removed. Reintroduction requires a
fresh implementation using CUBE_M16/M32, CUBE_N8, explicit accumulator/output
types, reviewed Shared publication, and an end-to-end oracle.

This archive also contains the former one-level `kernels/fa/`,
`kernels/matmul/`, their test drivers, and the DeepSeek multilayer-recompute
GEMM. They were build-reachable but did not satisfy the released CELL,
accumulator, Bias, scale, or postprocess contracts; their active Makefile and
compile-list entries were removed.
