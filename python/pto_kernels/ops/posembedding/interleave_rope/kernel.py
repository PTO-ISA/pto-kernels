"""Constrained PTO-DSL seed for interleave_rope on 910B.

The current seed keeps the full op contract on NPU, but splits the work into:

1. NPU-side interleave preprocessing via reshape + transpose
2. PTODSL rotary-half compute on the interleaved tensor

This keeps the public benchmark path aligned with the baseline input while
recording the real remaining gap: PTODSL still lacks a native vector-shuffle /
GatherMask-style surface for the upstream interleave stage.
"""

from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.ops.posembedding.interleave_rope.runtime import interleave_npu
from pto_kernels.utils.tuning import tuned_int


const = pto.const

D = 64
HALF_D = D // 2


def _block_dim() -> int:
    return tuned_int("PTO_INTERLEAVE_ROPE_BLOCK_DIM", 4, valid_values=(1, 2, 4, 8))


def _total_rows() -> int:
    return tuned_int("PTO_INTERLEAVE_ROPE_TOTAL_ROWS", 32, valid_values=(32, 64))


def _meta_data() -> dict[str, object]:
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


def _build_stage_kernel(*, output_dir):
    total_rows = _total_rows()
    block_dim = _block_dim()

    @jit(
        meta_data=_meta_data,
        output_dir=output_dir,
        block_dim=block_dim,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def interleave_rope_fp16_bnsd_stage(
        out_ptr: "ptr",
        q_ptr: "ptr",
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

        tv_out = pto.as_tensor(tensor, ptr=out_ptr, shape=[cRows, cD], strides=[cD, c1])
        tv_q = pto.as_tensor(tensor, ptr=q_ptr, shape=[cRows, cD], strides=[cD, c1])
        tv_cos = pto.as_tensor(tensor, ptr=cos_ptr, shape=[cRows, cD], strides=[cD, c1])
        tv_sin = pto.as_tensor(tensor, ptr=sin_ptr, shape=[cRows, cD], strides=[cD, c1])

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cRows, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cRows)

            tb_q = pto.alloc_tile(tile_full)
            tb_cos = pto.alloc_tile(tile_full)
            tb_sin = pto.alloc_tile(tile_full)
            tb_out = pto.alloc_tile(tile_full)
            tb_tmp0 = pto.alloc_tile(tile_half)
            tb_tmp1 = pto.alloc_tile(tile_half)
            tb_zero = pto.alloc_tile(tile_half)

            if rows == cRows:
                for row_idx in range(row_start, row_end, c1):
                    sv_q = pto.slice_view(row_view, source=tv_q, offsets=[row_idx, c0], sizes=[c1, cD])
                    sv_cos = pto.slice_view(row_view, source=tv_cos, offsets=[row_idx, c0], sizes=[c1, cD])
                    sv_sin = pto.slice_view(row_view, source=tv_sin, offsets=[row_idx, c0], sizes=[c1, cD])
                    sv_out = pto.slice_view(row_view, source=tv_out, offsets=[row_idx, c0], sizes=[c1, cD])

                    pto.load(sv_q, tb_q)
                    pto.load(sv_cos, tb_cos)
                    pto.load(sv_sin, tb_sin)

                    q1 = pto.subset(tb_q, [c0, c0], [1, HALF_D])
                    q2 = pto.subset(tb_q, [c0, cHalfD], [1, HALF_D])
                    cos1 = pto.subset(tb_cos, [c0, c0], [1, HALF_D])
                    cos2 = pto.subset(tb_cos, [c0, cHalfD], [1, HALF_D])
                    sin1 = pto.subset(tb_sin, [c0, c0], [1, HALF_D])
                    sin2 = pto.subset(tb_sin, [c0, cHalfD], [1, HALF_D])
                    out1 = pto.subset(tb_out, [c0, c0], [1, HALF_D])
                    out2 = pto.subset(tb_out, [c0, cHalfD], [1, HALF_D])

                    pto.sub(cos1, cos1, tb_zero)

                    pto.mul(q1, cos1, tb_tmp0)
                    pto.sub(tb_zero, q2, tb_tmp1)
                    pto.mul(tb_tmp1, sin1, tb_tmp1)
                    pto.add(tb_tmp0, tb_tmp1, out1)

                    pto.mul(q2, cos2, tb_tmp0)
                    pto.mul(q1, sin2, tb_tmp1)
                    pto.add(tb_tmp0, tb_tmp1, out2)

                    pto.store(tb_out, sv_out)

    return interleave_rope_fp16_bnsd_stage


class InterleaveRopeWrapper:
    def __init__(self, *, output_dir):
        self._output_dir = Path(output_dir)
        self._stage = _build_stage_kernel(output_dir=self._output_dir / "stage2_rotary")

    def _build(self):
        build = getattr(self._stage, "_build", None)
        if callable(build):
            build()

    def _artifact_paths(self):
        artifact_paths = getattr(self._stage, "_artifact_paths", None)
        return tuple(artifact_paths()) if callable(artifact_paths) else ()

    @property
    def library_path(self):
        return getattr(self._stage, "library_path", None)

    @property
    def output_dir(self):
        return str(self._output_dir)

    def __call__(self, out_ptr, x_ptr, cos_ptr, sin_ptr, rows_i32, stream_ptr=None):
        q = interleave_npu(x_ptr)
        self._stage(out_ptr, q, cos_ptr, sin_ptr, rows_i32, stream_ptr=stream_ptr)


def build_jit_wrapper(*, output_dir):
    return InterleaveRopeWrapper(output_dir=output_dir)
