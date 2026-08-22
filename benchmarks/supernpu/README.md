# SuperNPU benchmark suite

This subtree is the maintained SuperNPUBench workload collection inside
`PTO-ISA/pto-kernels`. Its source snapshot and exact provenance are recorded in
[`SOURCE.lock.json`](SOURCE.lock.json).

## Active architecture contract

- PTO ISA 0.58.3: exact released authority recorded in
  [`PTO_ISA.lock.json`](../../PTO_ISA.lock.json), including source commit/tree,
  encoding ABI, projection/content hashes, and instruction counts.
- Semantic engines: **VEC**, **TLSU**, **CUBE**, and **SFU**.
- VEC contains elementwise operations only; complex operations use SFU.
- Local `B.IOT` SizeCode 1..10 encodes 128 B..64 KiB per participating PE;
  Shared `B.IOS` SizeCode 1..12 encodes 128 B..256 KiB per participating PE.
- The exact PEMode mask decoder is `0000, 1000, 0100, 0010, 0001, 1100,
  1110, 1111`; zero is a strict no-op and other four-bit masks are illegal.
- Local CUBE primaries use persistent `CUBE_M16`, `CUBE_M32`, and `CUBE_N8`
  CELL layouts with explicit GM conversion at `TLOAD_CUBE`/`TSTORE_CUBE`.
- Matrix operations carry exactly one `B.FPATR`; its low fields include
  `TransA` and `TransB`, which are legal only for the corresponding Shared
  primary.
- Active assembly and intrinsic names must come from the v0.58 PTO/Linx
  specifications and the matching Linx-TileOP-API.

The eight retired body branches `B.EQ`, `B.NE`, `B.LT`, `B.GE`, `B.LTU`,
`B.GEU`, `B.Z`, and `B.NZ` are rejected from active sources. Historical files
under `status/legacy/` remain non-normative evidence.

Historical source, reports, and the pre-v0.58 embedded two-level API are isolated under
[`status/legacy/`](status/legacy/README.md) and are excluded from active checks.
The former `fa_2d_unroll_gmma.cpp` SPMD sketch is also archived there: it mixed
legacy matrix tiles, ordinary DMA, Vec accumulators, and `TMATMUL_FIXP` without
an executable result oracle, so it is not an active 0.58.3 workload.

## Layout

```text
benchmark/one-level-arch/  PTO tile-operation kernels and tests
microbenchmark/            scalar, VEC/SFU, TLSU, and CUBE microbenchmarks
docs/                      current workflow notes
status/legacy/             non-normative imported history
```

The Linx two-level implementation uses the separately versioned
`LinxISA/Linx-TileOP-API`; this repository does not carry a second copy of that
API as an architectural authority.

## Validation

The default repository check is lightweight and does not require an NPU:

```bash
python3 scripts/check_supernpu_v058.py
python3 scripts/check_pto_isa_0583.py
```

Compilation requires a LinxISA v0.58 compiler/sysroot and a matching
Linx-TileOP-API checkout. Set `COMPILER_DIR` to the compiler `bin/` directory
and `LINX_TILEOP_API_ROOT` to the TileOP API repository root. Set
`LINX_SYSROOT` to the matching musl sysroot containing libc++, libc++abi,
libunwind, and the Linx compiler runtime before running a benchmark Makefile.
The build fails closed when any of these three roots is omitted.

Full compiler, model, and hardware coverage remains an explicit release or
bring-up action; missing or skipped runs are not success.

The build consumes `LINX_TILEOP_API_ROOT` as an explicit checkout and does not
pin a temporary LLVM topic SHA. The final compiler identity belongs in the
superproject component lock after its reviewed leaf PR is merged.
