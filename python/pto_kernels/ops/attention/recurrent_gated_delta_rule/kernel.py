"""Blocked PTO skeleton for recurrent_gated_delta_rule on 910B/A3.

This kernel family is intentionally not implemented with scalar-loop fallback.
The upstream operator depends on tile-first recurrent state update primitives:
- BF16 state load/store with FP32 working accumulation
- state @ key matvec
- vector outer-product update into a state matrix
- ragged token-to-state mapping across sequence groups

PTODSL now exposes the tile-first `pto.gemv(...)` surface for the state-matvec
portion of that pattern, and the PTO stack also exposes column-broadcast binops
(`pto.col_expand_mul/sub/div`) that can be used to build the rank-1 update
without falling back to scalar loops. The checked-in migration slice remains
PTO-blocked until the stack grows the remaining recurrent integration pieces
after that:
- a checked tile-first outer-product state update path in the kernel
- ragged token-to-state writeback/mapping
"""


def build_jit_wrapper(*, output_dir):
    raise RuntimeError(
        "PTODSL/PTOAS tile-first recurrent state update surface is not available for recurrent_gated_delta_rule."
    )
