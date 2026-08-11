# SuperNPU benchmark suite

This subtree is the maintained SuperNPUBench workload collection inside
`PTO-ISA/pto-kernels`. Its source snapshot and exact provenance are recorded in
[`SOURCE.lock.json`](SOURCE.lock.json).

## Active architecture contract

- PTO-ISA and LinxISA: released v0.58 common tile-operation contract.
- Semantic engines: **VEC**, **TLSU**, **CUBE**, and **SFU**.
- VEC contains elementwise operations only; complex operations use SFU.
- Tile size is 128 B..8 KiB per participating PE.
- Active assembly and intrinsic names must come from the v0.58 PTO/Linx
  specifications and the matching Linx-TileOP-API.

Historical source, reports, and the pre-v0.58 embedded two-level API are isolated under
[`status/legacy/`](status/legacy/README.md) and are excluded from active checks.

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
```

Compilation requires a LinxISA v0.58 compiler/sysroot and a matching
Linx-TileOP-API checkout. Set `COMPILER_DIR` to the compiler `bin/` directory
and `LINX_TILEOP_API_ROOT` to the TileOP API repository root. Set
`LINX_SYSROOT` to the matching musl sysroot containing libc++, libc++abi,
libunwind, and the Linx compiler runtime before running a benchmark Makefile.
The build fails closed when any of these three roots is omitted.

Full compiler, model, and hardware coverage remains an explicit release or
bring-up action; missing or skipped runs are not success.
