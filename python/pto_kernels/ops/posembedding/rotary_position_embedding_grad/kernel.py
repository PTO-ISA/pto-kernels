"""Constrained PTO-DSL seed for rotary_position_embedding_grad on 910B."""

from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const

D = 128
HALF_D = D // 2


def _block_dim() -> int:
    return tuned_int("PTO_ROTARY_POSITION_EMBEDDING_GRAD_BLOCK_DIM", 4, valid_values=(1, 2, 4, 8))


def _total_rows() -> int:
    return tuned_int("PTO_ROTARY_POSITION_EMBEDDING_GRAD_TOTAL_ROWS", 64, valid_values=(64,))


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
    def rotary_position_embedding_grad(
        dx_ptr: "ptr",
        dcos_ptr: "ptr",
        dsin_ptr: "ptr",
        dy_ptr: "ptr",
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

            tv_dy = pto.as_tensor(tensor, ptr=dy_ptr, shape=[cRows, cD], strides=[cD, c1])
            tv_x = pto.as_tensor(tensor, ptr=x_ptr, shape=[cRows, cD], strides=[cD, c1])
            tv_cos = pto.as_tensor(tensor, ptr=cos_ptr, shape=[cRows, cD], strides=[cD, c1])
            tv_sin = pto.as_tensor(tensor, ptr=sin_ptr, shape=[cRows, cD], strides=[cD, c1])
            tv_dx = pto.as_tensor(tensor, ptr=dx_ptr, shape=[cRows, cD], strides=[cD, c1])
            tv_dcos = pto.as_tensor(tensor, ptr=dcos_ptr, shape=[cRows, cD], strides=[cD, c1])
            tv_dsin = pto.as_tensor(tensor, ptr=dsin_ptr, shape=[cRows, cD], strides=[cD, c1])

            if rows == cRows:
                tb_dy = pto.alloc_tile(tile_full)
                tb_x = pto.alloc_tile(tile_full)
                tb_cos = pto.alloc_tile(tile_full)
                tb_sin = pto.alloc_tile(tile_full)
                tb_dx = pto.alloc_tile(tile_full)
                tb_dcos = pto.alloc_tile(tile_full)
                tb_dsin = pto.alloc_tile(tile_full)
                tb_tmp0 = pto.alloc_tile(tile_half)
                tb_tmp1 = pto.alloc_tile(tile_half)
                tb_zero = pto.alloc_tile(tile_half)

                for row_idx in range(row_start, row_end, c1):
                    sv_dy = pto.slice_view(row_view, source=tv_dy, offsets=[row_idx, c0], sizes=[c1, cD])
                    sv_x = pto.slice_view(row_view, source=tv_x, offsets=[row_idx, c0], sizes=[c1, cD])
                    sv_cos = pto.slice_view(row_view, source=tv_cos, offsets=[row_idx, c0], sizes=[c1, cD])
                    sv_sin = pto.slice_view(row_view, source=tv_sin, offsets=[row_idx, c0], sizes=[c1, cD])
                    sv_dx = pto.slice_view(row_view, source=tv_dx, offsets=[row_idx, c0], sizes=[c1, cD])
                    sv_dcos = pto.slice_view(row_view, source=tv_dcos, offsets=[row_idx, c0], sizes=[c1, cD])
                    sv_dsin = pto.slice_view(row_view, source=tv_dsin, offsets=[row_idx, c0], sizes=[c1, cD])

                    pto.load(sv_dy, tb_dy)
                    pto.load(sv_x, tb_x)
                    pto.load(sv_cos, tb_cos)
                    pto.load(sv_sin, tb_sin)

                    dy1 = pto.subset(tb_dy, [c0, c0], [1, HALF_D])
                    dy2 = pto.subset(tb_dy, [c0, cHalfD], [1, HALF_D])
                    x1 = pto.subset(tb_x, [c0, c0], [1, HALF_D])
                    x2 = pto.subset(tb_x, [c0, cHalfD], [1, HALF_D])
                    cos1 = pto.subset(tb_cos, [c0, c0], [1, HALF_D])
                    cos2 = pto.subset(tb_cos, [c0, cHalfD], [1, HALF_D])
                    sin1 = pto.subset(tb_sin, [c0, c0], [1, HALF_D])
                    sin2 = pto.subset(tb_sin, [c0, cHalfD], [1, HALF_D])
                    dx1 = pto.subset(tb_dx, [c0, c0], [1, HALF_D])
                    dx2 = pto.subset(tb_dx, [c0, cHalfD], [1, HALF_D])
                    dsin1 = pto.subset(tb_dsin, [c0, c0], [1, HALF_D])
                    dsin2 = pto.subset(tb_dsin, [c0, cHalfD], [1, HALF_D])

                    pto.mul(tb_dy, tb_x, tb_dcos)

                    pto.mul(cos1, dy1, tb_tmp0)
                    pto.mul(sin2, dy2, tb_tmp1)
                    pto.add(tb_tmp0, tb_tmp1, dx1)

                    pto.mul(cos2, dy2, tb_tmp0)
                    pto.mul(sin1, dy1, tb_tmp1)
                    pto.sub(tb_tmp0, tb_tmp1, dx2)

                    pto.sub(x1, x1, tb_zero)
                    pto.sub(tb_zero, x2, tb_tmp0)
                    pto.mul(dy1, tb_tmp0, dsin1)
                    pto.mul(dy2, x1, dsin2)

                    pto.store(tb_dx, sv_dx)
                    pto.store(tb_dcos, sv_dcos)
                    pto.store(tb_dsin, sv_dsin)

    return rotary_position_embedding_grad


class RotaryPositionEmbeddingGradWrapper:
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

    def __call__(self, dy, x, cos, sin, rows, stream_ptr=None):
        dx = dy.new_empty(dy.shape)
        dcos = dy.new_empty(dy.shape)
        dsin = dy.new_empty(dy.shape)
        self._kernel(dx, dcos, dsin, dy, x, cos, sin, rows, stream_ptr=stream_ptr)
        return dx, dcos, dsin


def build_jit_wrapper(*, output_dir):
    return RotaryPositionEmbeddingGradWrapper(output_dir=output_dir)
