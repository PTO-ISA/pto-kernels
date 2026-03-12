"""Constrained PTO-DSL seed for apply_rotary_pos_emb on 910B.

The current seed keeps the flattened `[rows, D]` contract for the validated
`TND` and `BSND` half-mode cases, but the execution shape is now closer to the
upstream ops-transformer kernel:

- each core owns one contiguous chunk of rows
- query, key, cos, and sin are loaded once per chunk
- the rotation is applied over full `[chunk_rows, half_d]` vector tiles

The remaining gap is the upstream double-buffer queue pipeline. PTO source stays
sync-free and relies on PTOAS autosync insertion.
"""

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const

D = 128
HALF_D = D // 2
TOTAL_ROWS = 64


def _block_dim() -> int:
    return tuned_int("PTO_APPLY_ROTARY_BLOCK_DIM", 4, valid_values=(1, 2, 4, 8))


def _chunk_rows(block_dim: int) -> int:
    if TOTAL_ROWS % block_dim != 0:
        raise ValueError(f"apply_rotary_pos_emb seed requires TOTAL_ROWS={TOTAL_ROWS} to be divisible by block_dim={block_dim}")
    return TOTAL_ROWS // block_dim


def _meta_data(block_dim: int):
    dtype = pto.float16
    ptr = pto.PtrType(dtype)
    i32 = pto.int32

    tensor = pto.TensorType(rank=2, dtype=dtype)
    row_view = pto.SubTensorType(shape=[1, D], dtype=dtype)

    tile_full = pto.TileType(shape=[1, D], dtype=dtype, memory_space="VEC")
    tile_half = pto.TileType(shape=[1, HALF_D], dtype=dtype, memory_space="VEC")

    return {
        "ptr": ptr,
        "i32": i32,
        "tensor": tensor,
        "row_view": row_view,
        "tile_full": tile_full,
        "tile_half": tile_half,
    }


def build_jit_wrapper(*, output_dir):
    block_dim = _block_dim()
    chunk_rows = _chunk_rows(block_dim)

    @jit(
        meta_data=lambda: _meta_data(block_dim),
        output_dir=output_dir,
        block_dim=block_dim,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def apply_rotary_pos_emb_half_fp16_rows(
        query_ptr: "ptr",
        key_ptr: "ptr",
        cos_ptr: "ptr",
        sin_ptr: "ptr",
        rows_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cD = const(D)
        cHalfD = const(HALF_D)
        cRows = const(TOTAL_ROWS)
        cChunkRows = const(chunk_rows)

        rows = pto.index_cast(rows_i32)

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            row_start = bid * cChunkRows

            tv_query = pto.as_tensor(
                tensor,
                ptr=query_ptr,
                shape=[cRows, cD],
                strides=[cD, c1],
            )
            tv_key = pto.as_tensor(
                tensor,
                ptr=key_ptr,
                shape=[cRows, cD],
                strides=[cD, c1],
            )
            tv_cos = pto.as_tensor(
                tensor,
                ptr=cos_ptr,
                shape=[cRows, cD],
                strides=[cD, c1],
            )
            tv_sin = pto.as_tensor(
                tensor,
                ptr=sin_ptr,
                shape=[cRows, cD],
                strides=[cD, c1],
            )

            if rows == cRows:
                tb_query = pto.alloc_tile(tile_full)
                tb_key = pto.alloc_tile(tile_full)
                tb_cos = pto.alloc_tile(tile_full)
                tb_sin = pto.alloc_tile(tile_full)
                tb_query_out = pto.alloc_tile(tile_full)
                tb_key_out = pto.alloc_tile(tile_full)
                tb_tmp0 = pto.alloc_tile(tile_half)
                tb_tmp1 = pto.alloc_tile(tile_half)
                tb_zero = pto.alloc_tile(tile_half)

                for row_i in range(c0, cChunkRows, c1):
                    row_idx = row_start + row_i

                    sv_query = pto.slice_view(
                        row_view,
                        source=tv_query,
                        offsets=[row_idx, c0],
                        sizes=[c1, cD],
                    )
                    sv_key = pto.slice_view(
                        row_view,
                        source=tv_key,
                        offsets=[row_idx, c0],
                        sizes=[c1, cD],
                    )
                    sv_cos = pto.slice_view(
                        row_view,
                        source=tv_cos,
                        offsets=[row_idx, c0],
                        sizes=[c1, cD],
                    )
                    sv_sin = pto.slice_view(
                        row_view,
                        source=tv_sin,
                        offsets=[row_idx, c0],
                        sizes=[c1, cD],
                    )

                    pto.load(sv_query, tb_query)
                    pto.load(sv_key, tb_key)
                    pto.load(sv_cos, tb_cos)
                    pto.load(sv_sin, tb_sin)

                    q1 = pto.subset(tb_query, [c0, c0], [1, HALF_D])
                    q2 = pto.subset(tb_query, [c0, cHalfD], [1, HALF_D])
                    k1 = pto.subset(tb_key, [c0, c0], [1, HALF_D])
                    k2 = pto.subset(tb_key, [c0, cHalfD], [1, HALF_D])
                    cos1 = pto.subset(tb_cos, [c0, c0], [1, HALF_D])
                    cos2 = pto.subset(tb_cos, [c0, cHalfD], [1, HALF_D])
                    sin1 = pto.subset(tb_sin, [c0, c0], [1, HALF_D])
                    sin2 = pto.subset(tb_sin, [c0, cHalfD], [1, HALF_D])
                    q_out1 = pto.subset(tb_query_out, [c0, c0], [1, HALF_D])
                    q_out2 = pto.subset(tb_query_out, [c0, cHalfD], [1, HALF_D])
                    k_out1 = pto.subset(tb_key_out, [c0, c0], [1, HALF_D])
                    k_out2 = pto.subset(tb_key_out, [c0, cHalfD], [1, HALF_D])

                    pto.sub(cos1, cos1, tb_zero)

                    pto.mul(q1, cos1, tb_tmp0)
                    pto.sub(tb_zero, q2, tb_tmp1)
                    pto.mul(tb_tmp1, sin1, tb_tmp1)
                    pto.add(tb_tmp0, tb_tmp1, q_out1)
                    pto.mul(q2, cos2, tb_tmp0)
                    pto.mul(q1, sin2, tb_tmp1)
                    pto.add(tb_tmp0, tb_tmp1, q_out2)

                    pto.mul(k1, cos1, tb_tmp0)
                    pto.sub(tb_zero, k2, tb_tmp1)
                    pto.mul(tb_tmp1, sin1, tb_tmp1)
                    pto.add(tb_tmp0, tb_tmp1, k_out1)
                    pto.mul(k2, cos2, tb_tmp0)
                    pto.mul(k1, sin2, tb_tmp1)
                    pto.add(tb_tmp0, tb_tmp1, k_out2)

                    pto.store(tb_query_out, sv_query)
                    pto.store(tb_key_out, sv_key)

    return apply_rotary_pos_emb_half_fp16_rows
