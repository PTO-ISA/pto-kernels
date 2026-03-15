"""Constrained PTO-DSL seed for rotary_position_embedding on 910B."""

from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const

D = 128
HALF_D = D // 2


def _block_dim() -> int:
    return tuned_int("PTO_ROTARY_POSITION_EMBEDDING_BLOCK_DIM", 4, valid_values=(1, 2, 4, 8))


def _total_rows() -> int:
    return tuned_int("PTO_ROTARY_POSITION_EMBEDDING_TOTAL_ROWS", 64, valid_values=(64,))


def _meta_data():
    dtype = pto.float16
    ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)
    row_view = pto.SubTensorType(shape=[1, D], dtype=dtype)
    tile_full = pto.TileType(shape=[1, D], dtype=dtype, memory_space="VEC")
    tile_half = pto.TileType(shape=[1, HALF_D], dtype=dtype, memory_space="VEC")
    return {
        "ptr": ptr,
        "i32": pto.int32,
        "tensor": tensor,
        "row_view": row_view,
        "tile_full": tile_full,
        "tile_half": tile_half,
    }


def _build_kernel(*, output_dir):
    total_rows = _total_rows()
    block_dim = _block_dim()

    @jit(
        meta_data=_meta_data,
        output_dir=output_dir,
        block_dim=block_dim,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def rotary_position_embedding(
        out_ptr: "ptr",
        x_ptr: "ptr",
        cos_ptr: "ptr",
        sin_ptr: "ptr",
        rows_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cD = const(D)
        cHalfD = const(HALF_D)
        cRows = const(total_rows)

        rows = pto.index_cast(rows_i32)

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cRows, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cRows)

            tv_x = pto.as_tensor(tensor, ptr=x_ptr, shape=[cRows, cD], strides=[cD, c1])
            tv_cos = pto.as_tensor(tensor, ptr=cos_ptr, shape=[cRows, cD], strides=[cD, c1])
            tv_sin = pto.as_tensor(tensor, ptr=sin_ptr, shape=[cRows, cD], strides=[cD, c1])
            tv_out = pto.as_tensor(tensor, ptr=out_ptr, shape=[cRows, cD], strides=[cD, c1])

            if rows == cRows:
                tb_x = pto.alloc_tile(tile_full)
                tb_cos = pto.alloc_tile(tile_full)
                tb_sin = pto.alloc_tile(tile_full)
                tb_out = pto.alloc_tile(tile_full)
                tb_tmp0 = pto.alloc_tile(tile_half)
                tb_tmp1 = pto.alloc_tile(tile_half)
                tb_zero = pto.alloc_tile(tile_half)

                for row_idx in range(row_start, row_end, c1):
                    sv_x = pto.slice_view(row_view, source=tv_x, offsets=[row_idx, c0], sizes=[c1, cD])
                    sv_cos = pto.slice_view(row_view, source=tv_cos, offsets=[row_idx, c0], sizes=[c1, cD])
                    sv_sin = pto.slice_view(row_view, source=tv_sin, offsets=[row_idx, c0], sizes=[c1, cD])
                    sv_out = pto.slice_view(row_view, source=tv_out, offsets=[row_idx, c0], sizes=[c1, cD])

                    pto.load(sv_x, tb_x)
                    pto.load(sv_cos, tb_cos)
                    pto.load(sv_sin, tb_sin)

                    x1 = pto.subset(tb_x, [c0, c0], [1, HALF_D])
                    x2 = pto.subset(tb_x, [c0, cHalfD], [1, HALF_D])
                    cos1 = pto.subset(tb_cos, [c0, c0], [1, HALF_D])
                    cos2 = pto.subset(tb_cos, [c0, cHalfD], [1, HALF_D])
                    sin1 = pto.subset(tb_sin, [c0, c0], [1, HALF_D])
                    sin2 = pto.subset(tb_sin, [c0, cHalfD], [1, HALF_D])
                    out1 = pto.subset(tb_out, [c0, c0], [1, HALF_D])
                    out2 = pto.subset(tb_out, [c0, cHalfD], [1, HALF_D])

                    pto.sub(cos1, cos1, tb_zero)

                    pto.mul(x1, cos1, tb_tmp0)
                    pto.sub(tb_zero, x2, tb_tmp1)
                    pto.mul(tb_tmp1, sin1, tb_tmp1)
                    pto.add(tb_tmp0, tb_tmp1, out1)
                    pto.mul(x2, cos2, tb_tmp0)
                    pto.mul(x1, sin2, tb_tmp1)
                    pto.add(tb_tmp0, tb_tmp1, out2)

                    pto.store(tb_out, sv_out)

    return rotary_position_embedding


class RotaryPositionEmbeddingWrapper:
    def __init__(self, *, output_dir):
        self._output_dir = Path(output_dir)
        self._kernel = _build_kernel(output_dir=self._output_dir)

    def _build(self):
        self._kernel._build()

    def _artifact_paths(self):
        return tuple(self._kernel._artifact_paths())

    @property
    def output_dir(self):
        return str(self._output_dir)

    def __call__(self, out, x, cos, sin, rows, stream_ptr=None):
        self._kernel(out, x, cos, sin, rows, stream_ptr=stream_ptr)


def build_jit_wrapper(*, output_dir):
    return RotaryPositionEmbeddingWrapper(output_dir=output_dir)
