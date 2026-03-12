"""Constrained PTO-DSL seed for apply_rotary_pos_emb on 910B.

The PTO kernel operates over a flattened contiguous `[rows, D]` view, which
lets the current seed cover both:

- `TND` with `rows = tokens * heads`
- `BSND` with `rows = batch * seq * heads`

Current limits remain:

- dtype = float16
- query/key heads fixed to 1
- head_dim fixed to 128

Additional rotary modes and broader head/layout semantics remain follow-up
work.
"""

from ptodsl import jit, pto, tile
from ptodsl import scalar as s
from pto_kernels.utils.tuning import tuned_int


const = s.const

D = 128
HALF_D = D // 2


def _meta_data():
    dtype = pto.float16
    ptr = pto.PtrType(dtype)
    i32 = pto.int32

    tensor = pto.TensorType(rank=2, dtype=dtype)
    row_view = pto.SubTensorType(shape=[1, D], dtype=dtype)

    tile_full = pto.TileBufType(shape=[1, D], dtype=dtype, memory_space="VEC")
    tile_half = pto.TileBufType(shape=[1, HALF_D], dtype=dtype, memory_space="VEC")

    return {
        "ptr": ptr,
        "i32": i32,
        "tensor": tensor,
        "row_view": row_view,
        "tile_full": tile_full,
        "tile_half": tile_half,
    }


def build_jit_wrapper(*, output_dir):
    @jit(
        meta_data=_meta_data,
        output_dir=output_dir,
        block_dim=tuned_int("PTO_APPLY_ROTARY_BLOCK_DIM", 4, valid_values=(1, 2, 4, 8)),
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

        rows = s.index_cast(rows_i32)

        with pto.vector_section():
            cid = pto.get_block_idx()
            sub_bid = pto.get_subblock_idx()
            sub_bnum = pto.get_subblock_num()
            num_blocks = pto.get_block_num()

            vid = s.index_cast(cid * sub_bnum + sub_bid)
            num_cores = s.index_cast(num_blocks * sub_bnum)

            rows_per_core = s.ceil_div(rows, num_cores)
            row_start = vid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, rows)
            num_rows = row_end - row_start

            tv_query = pto.as_tensor(
                tensor,
                ptr=query_ptr,
                shape=[rows, cD],
                strides=[cD, c1],
            )
            tv_key = pto.as_tensor(
                tensor,
                ptr=key_ptr,
                shape=[rows, cD],
                strides=[cD, c1],
            )
            tv_cos = pto.as_tensor(
                tensor,
                ptr=cos_ptr,
                shape=[rows, cD],
                strides=[cD, c1],
            )
            tv_sin = pto.as_tensor(
                tensor,
                ptr=sin_ptr,
                shape=[rows, cD],
                strides=[cD, c1],
            )

            with pto.if_context(num_rows > c0):
                tb_query = pto.alloc_tile(tile_full)
                tb_key = pto.alloc_tile(tile_full)
                tb_cos = pto.alloc_tile(tile_full)
                tb_sin = pto.alloc_tile(tile_full)
                tb_query_out = pto.alloc_tile(tile_full)
                tb_key_out = pto.alloc_tile(tile_full)
                tb_tmp0 = pto.alloc_tile(tile_half)
                tb_tmp1 = pto.alloc_tile(tile_half)
                tb_zero = pto.alloc_tile(tile_half)

                for row_i in pto.range(c0, num_rows, c1):
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

                    q1 = tile.subset(tb_query, [c0, c0], [1, HALF_D])
                    q2 = tile.subset(tb_query, [c0, cHalfD], [1, HALF_D])
                    k1 = tile.subset(tb_key, [c0, c0], [1, HALF_D])
                    k2 = tile.subset(tb_key, [c0, cHalfD], [1, HALF_D])
                    cos1 = tile.subset(tb_cos, [c0, c0], [1, HALF_D])
                    cos2 = tile.subset(tb_cos, [c0, cHalfD], [1, HALF_D])
                    sin1 = tile.subset(tb_sin, [c0, c0], [1, HALF_D])
                    sin2 = tile.subset(tb_sin, [c0, cHalfD], [1, HALF_D])
                    q_out1 = tile.subset(tb_query_out, [c0, c0], [1, HALF_D])
                    q_out2 = tile.subset(tb_query_out, [c0, cHalfD], [1, HALF_D])
                    k_out1 = tile.subset(tb_key_out, [c0, c0], [1, HALF_D])
                    k_out2 = tile.subset(tb_key_out, [c0, cHalfD], [1, HALF_D])

                    tile.sub(cos1, cos1, tb_zero)

                    tile.mul(q1, cos1, tb_tmp0)
                    tile.sub(tb_zero, q2, tb_tmp1)
                    tile.mul(tb_tmp1, sin1, tb_tmp1)
                    tile.add(tb_tmp0, tb_tmp1, q_out1)
                    tile.mul(q2, cos2, tb_tmp0)
                    tile.mul(q1, sin2, tb_tmp1)
                    tile.add(tb_tmp0, tb_tmp1, q_out2)

                    tile.mul(k1, cos1, tb_tmp0)
                    tile.sub(tb_zero, k2, tb_tmp1)
                    tile.mul(tb_tmp1, sin1, tb_tmp1)
                    tile.add(tb_tmp0, tb_tmp1, k_out1)
                    tile.mul(k2, cos2, tb_tmp0)
                    tile.mul(k1, sin2, tb_tmp1)
                    tile.add(tb_tmp0, tb_tmp1, k_out2)

                    pto.store(tb_query_out, sv_query)
                    pto.store(tb_key_out, sv_key)

    return apply_rotary_pos_emb_half_fp16_rows
