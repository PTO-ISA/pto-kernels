"""Constrained PTO-DSL seed for rope_with_sin_cos_cache on 910B."""

from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const

D = 64
HALF_D = D // 2
COS_SIN_D = D * 2


def _block_dim() -> int:
    return tuned_int("PTO_ROPE_WITH_SIN_COS_CACHE_BLOCK_DIM", 4, valid_values=(1, 2, 4, 8))


def _total_rows() -> int:
    return tuned_int("PTO_ROPE_WITH_SIN_COS_CACHE_TOTAL_ROWS", 2, valid_values=(2, 4))


def _meta_data() -> dict[str, object]:
    fp16 = pto.float16
    ptr_fp16 = pto.PtrType(fp16)
    tensor_row = pto.TensorType(rank=2, dtype=fp16)
    tensor_cache = pto.TensorType(rank=2, dtype=fp16)
    row_view = pto.SubTensorType(shape=[1, D], dtype=fp16)
    cache_view = pto.SubTensorType(shape=[1, COS_SIN_D], dtype=fp16)
    tile_full = pto.TileType(shape=[1, D], dtype=fp16, memory_space="VEC")
    tile_cache = pto.TileType(shape=[1, COS_SIN_D], dtype=fp16, memory_space="VEC")
    tile_half = pto.TileType(shape=[1, HALF_D], dtype=fp16, memory_space="VEC")
    return {
        "ptr_fp16": ptr_fp16,
        "i32": pto.int32,
        "tensor_row": tensor_row,
        "tensor_cache": tensor_cache,
        "row_view": row_view,
        "cache_view": cache_view,
        "tile_full": tile_full,
        "tile_cache": tile_cache,
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
    def rope_with_sin_cos_cache(
        query_out_ptr: "ptr_fp16",
        key_out_ptr: "ptr_fp16",
        query_ptr: "ptr_fp16",
        key_ptr: "ptr_fp16",
        cache_ptr: "ptr_fp16",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cRows = const(total_rows)
        cHalf = const(HALF_D)
        cD = const(D)
        cCacheD = const(COS_SIN_D)

        query_in = pto.as_tensor(tensor_row, ptr=query_ptr, shape=[cRows, cD], strides=[cD, c1])
        key_in = pto.as_tensor(tensor_row, ptr=key_ptr, shape=[cRows, cD], strides=[cD, c1])
        query_out = pto.as_tensor(tensor_row, ptr=query_out_ptr, shape=[cRows, cD], strides=[cD, c1])
        key_out = pto.as_tensor(tensor_row, ptr=key_out_ptr, shape=[cRows, cD], strides=[cD, c1])
        cache = pto.as_tensor(tensor_cache, ptr=cache_ptr, shape=[cRows, cCacheD], strides=[cCacheD, c1])

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cRows, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cRows)

            query_tile = pto.alloc_tile(tile_full)
            key_tile = pto.alloc_tile(tile_full)
            cache_tile = pto.alloc_tile(tile_cache)
            tmp0 = pto.alloc_tile(tile_half)
            tmp1 = pto.alloc_tile(tile_half)
            zero_half = pto.alloc_tile(tile_half)
            q_out_tile = pto.alloc_tile(tile_full)
            k_out_tile = pto.alloc_tile(tile_full)

            for row_idx in range(row_start, row_end, c1):
                query_in_view = pto.slice_view(row_view, source=query_in, offsets=[row_idx, c0], sizes=[c1, cD])
                key_in_view = pto.slice_view(row_view, source=key_in, offsets=[row_idx, c0], sizes=[c1, cD])
                query_out_view = pto.slice_view(row_view, source=query_out, offsets=[row_idx, c0], sizes=[c1, cD])
                key_out_view = pto.slice_view(row_view, source=key_out, offsets=[row_idx, c0], sizes=[c1, cD])
                cache_view_cur = pto.slice_view(cache_view, source=cache, offsets=[row_idx, c0], sizes=[c1, cCacheD])

                pto.load(query_in_view, query_tile)
                pto.load(key_in_view, key_tile)
                pto.load(cache_view_cur, cache_tile)

                cos_tile = pto.subset(cache_tile, [c0, c0], [1, D])
                sin_tile = pto.subset(cache_tile, [c0, cD], [1, D])

                cos1 = pto.subset(cos_tile, [c0, c0], [1, HALF_D])
                cos2 = pto.subset(cos_tile, [c0, cHalf], [1, HALF_D])
                sin1 = pto.subset(sin_tile, [c0, c0], [1, HALF_D])
                sin2 = pto.subset(sin_tile, [c0, cHalf], [1, HALF_D])

                q_x1 = pto.subset(query_tile, [c0, c0], [1, HALF_D])
                q_x2 = pto.subset(query_tile, [c0, cHalf], [1, HALF_D])
                k_x1 = pto.subset(key_tile, [c0, c0], [1, HALF_D])
                k_x2 = pto.subset(key_tile, [c0, cHalf], [1, HALF_D])

                q_out1 = pto.subset(q_out_tile, [c0, c0], [1, HALF_D])
                q_out2 = pto.subset(q_out_tile, [c0, cHalf], [1, HALF_D])
                k_out1 = pto.subset(k_out_tile, [c0, c0], [1, HALF_D])
                k_out2 = pto.subset(k_out_tile, [c0, cHalf], [1, HALF_D])

                pto.sub(cos1, cos1, zero_half)

                pto.mul(q_x1, cos1, tmp0)
                pto.sub(zero_half, q_x2, tmp1)
                pto.mul(tmp1, sin1, tmp1)
                pto.add(tmp0, tmp1, q_out1)
                pto.mul(q_x2, cos2, tmp0)
                pto.mul(q_x1, sin2, tmp1)
                pto.add(tmp0, tmp1, q_out2)

                pto.mul(k_x1, cos1, tmp0)
                pto.sub(zero_half, k_x2, tmp1)
                pto.mul(tmp1, sin1, tmp1)
                pto.add(tmp0, tmp1, k_out1)
                pto.mul(k_x2, cos2, tmp0)
                pto.mul(k_x1, sin2, tmp1)
                pto.add(tmp0, tmp1, k_out2)

                pto.store(q_out_tile, query_out_view)
                pto.store(k_out_tile, key_out_view)

    return rope_with_sin_cos_cache


class RopeWithSinCosCacheWrapper:
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

    def __call__(self, query_out, key_out, query, key, cos_sin_cache, stream_ptr=None):
        self._kernel(
            query_out,
            key_out,
            query,
            key,
            cos_sin_cache,
            stream_ptr=stream_ptr,
        )


def build_jit_wrapper(*, output_dir):
    return RopeWithSinCosCacheWrapper(output_dir=output_dir)
