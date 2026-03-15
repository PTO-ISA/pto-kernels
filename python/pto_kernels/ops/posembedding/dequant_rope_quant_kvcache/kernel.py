"""Constrained PTO-DSL seed for dequant_rope_quant_kvcache on 910B."""

from pathlib import Path

import torch

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const

D = 64
HALF_D = D // 2
SIZE_SPLITS = (64, 64, 64)
TOTAL_D = sum(SIZE_SPLITS)


def _block_dim() -> int:
    return tuned_int("PTO_ROPE_KVCACHE_BLOCK_DIM", 4, valid_values=(1, 2, 4, 8))


def _total_rows() -> int:
    return tuned_int("PTO_ROPE_KVCACHE_TOTAL_ROWS", 2, valid_values=(2, 4))


def _rotary_meta_data() -> dict[str, object]:
    fp16 = pto.float16
    fp32 = pto.float32
    i32 = pto.int32
    ptr_fp16 = pto.PtrType(fp16)
    ptr_fp32 = pto.PtrType(fp32)
    ptr_i32 = pto.PtrType(i32)
    tensor_in = pto.TensorType(rank=2, dtype=i32)
    tensor_row = pto.TensorType(rank=2, dtype=fp16)
    tensor_scale_full = pto.TensorType(rank=2, dtype=fp32)
    in_view = pto.SubTensorType(shape=[1, TOTAL_D], dtype=i32)
    row_view = pto.SubTensorType(shape=[1, D], dtype=fp16)
    scale_view = pto.SubTensorType(shape=[1, TOTAL_D], dtype=fp32)
    tile_in = pto.TileType(shape=[1, TOTAL_D], dtype=i32, memory_space="VEC")
    tile_fp32_full = pto.TileType(shape=[1, TOTAL_D], dtype=fp32, memory_space="VEC")
    tile_fp16_full = pto.TileType(shape=[1, TOTAL_D], dtype=fp16, memory_space="VEC")
    tile_full = pto.TileType(shape=[1, D], dtype=fp16, memory_space="VEC")
    tile_half = pto.TileType(shape=[1, HALF_D], dtype=fp16, memory_space="VEC")
    return {
        "ptr_fp16": ptr_fp16,
        "ptr_fp32": ptr_fp32,
        "ptr_i32": ptr_i32,
        "i32": pto.int32,
        "tensor_in": tensor_in,
        "tensor_row": tensor_row,
        "tensor_scale_full": tensor_scale_full,
        "in_view": in_view,
        "row_view": row_view,
        "scale_view": scale_view,
        "tile_in": tile_in,
        "tile_fp32_full": tile_fp32_full,
        "tile_fp16_full": tile_fp16_full,
        "tile_full": tile_full,
        "tile_half": tile_half,
    }


def _cache_meta_data() -> dict[str, object]:
    fp16 = pto.float16
    i8 = pto.int8
    ptr_fp16 = pto.PtrType(fp16)
    ptr_i8 = pto.PtrType(i8)
    tensor_row = pto.TensorType(rank=2, dtype=fp16)
    tensor_cache = pto.TensorType(rank=2, dtype=i8)
    row_view = pto.SubTensorType(shape=[1, D], dtype=fp16)
    cache_view = pto.SubTensorType(shape=[1, D], dtype=i8)
    tile_full = pto.TileType(shape=[1, D], dtype=fp16, memory_space="VEC")
    tile_full_i8 = pto.TileType(shape=[1, D], dtype=i8, memory_space="VEC")
    return {
        "ptr_fp16": ptr_fp16,
        "ptr_i8": ptr_i8,
        "i32": pto.int32,
        "tensor_row": tensor_row,
        "tensor_cache": tensor_cache,
        "row_view": row_view,
        "cache_view": cache_view,
        "tile_full": tile_full,
        "tile_full_i8": tile_full_i8,
    }


def _build_rotary_stage(*, output_dir):
    total_rows = _total_rows()
    block_dim = _block_dim()

    @jit(
        meta_data=_rotary_meta_data,
        output_dir=output_dir,
        block_dim=block_dim,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def dequant_rope_quant_kvcache_rotary_stage(
        q_ptr: "ptr_fp16",
        k_ptr: "ptr_fp16",
        v_ptr: "ptr_fp16",
        x_ptr: "ptr_i32",
        weight_scale_ptr: "ptr_fp32",
        cos_ptr: "ptr_fp16",
        sin_ptr: "ptr_fp16",
        rows_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cRows = const(total_rows)
        c64 = const(D)
        c128 = const(128)
        cTotalD = const(TOTAL_D)
        cHalf = const(HALF_D)

        rows = pto.index_cast(rows_i32)
        q_tensor = pto.as_tensor(tensor_row, ptr=q_ptr, shape=[cRows, c64], strides=[c64, c1])
        k_tensor = pto.as_tensor(tensor_row, ptr=k_ptr, shape=[cRows, c64], strides=[c64, c1])
        v_tensor = pto.as_tensor(tensor_row, ptr=v_ptr, shape=[cRows, c64], strides=[c64, c1])
        x_tensor = pto.as_tensor(tensor_in, ptr=x_ptr, shape=[cRows, cTotalD], strides=[cTotalD, c1])
        scale_tensor = pto.as_tensor(tensor_scale_full, ptr=weight_scale_ptr, shape=[c1, cTotalD], strides=[cTotalD, c1])
        cos_tensor = pto.as_tensor(tensor_row, ptr=cos_ptr, shape=[cRows, c64], strides=[c64, c1])
        sin_tensor = pto.as_tensor(tensor_row, ptr=sin_ptr, shape=[cRows, c64], strides=[c64, c1])

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cRows, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cRows)

            x_i32 = pto.alloc_tile(tile_in)
            scale_full = pto.alloc_tile(tile_fp32_full)
            deq_full = pto.alloc_tile(tile_fp32_full)
            deq_half = pto.alloc_tile(tile_fp16_full)
            cos_tile = pto.alloc_tile(tile_full)
            sin_tile = pto.alloc_tile(tile_full)
            q_tile = pto.alloc_tile(tile_full)
            k_tile = pto.alloc_tile(tile_full)
            v_tile = pto.alloc_tile(tile_full)
            tmp0 = pto.alloc_tile(tile_half)
            tmp1 = pto.alloc_tile(tile_half)
            zero_half = pto.alloc_tile(tile_half)
            if rows == cRows:
                for row_idx in range(row_start, row_end, c1):
                    x_view = pto.slice_view(in_view, source=x_tensor, offsets=[row_idx, c0], sizes=[c1, cTotalD])
                    scale_view_full = pto.slice_view(scale_view, source=scale_tensor, offsets=[c0, c0], sizes=[c1, cTotalD])
                    cos_view = pto.slice_view(row_view, source=cos_tensor, offsets=[row_idx, c0], sizes=[c1, c64])
                    sin_view = pto.slice_view(row_view, source=sin_tensor, offsets=[row_idx, c0], sizes=[c1, c64])
                    q_view = pto.slice_view(row_view, source=q_tensor, offsets=[row_idx, c0], sizes=[c1, c64])
                    k_view = pto.slice_view(row_view, source=k_tensor, offsets=[row_idx, c0], sizes=[c1, c64])
                    v_view = pto.slice_view(row_view, source=v_tensor, offsets=[row_idx, c0], sizes=[c1, c64])

                    pto.load(x_view, x_i32)
                    pto.load(scale_view_full, scale_full)
                    pto.cvt(x_i32, deq_full)
                    pto.mul(deq_full, scale_full, deq_full)
                    pto.cvt(deq_full, deq_half)
                    pto.load(cos_view, cos_tile)
                    pto.load(sin_view, sin_tile)

                    q_raw = pto.subset(deq_half, [c0, c0], [1, D])
                    k_raw = pto.subset(deq_half, [c0, c64], [1, D])
                    v_raw = pto.subset(deq_half, [c0, c128], [1, D])

                    q_x1 = pto.subset(q_raw, [c0, c0], [1, HALF_D])
                    q_x2 = pto.subset(q_raw, [c0, cHalf], [1, HALF_D])
                    k_x1 = pto.subset(k_raw, [c0, c0], [1, HALF_D])
                    k_x2 = pto.subset(k_raw, [c0, cHalf], [1, HALF_D])
                    cos1 = pto.subset(cos_tile, [c0, c0], [1, HALF_D])
                    cos2 = pto.subset(cos_tile, [c0, cHalf], [1, HALF_D])
                    sin1 = pto.subset(sin_tile, [c0, c0], [1, HALF_D])
                    sin2 = pto.subset(sin_tile, [c0, cHalf], [1, HALF_D])
                    q_out1 = pto.subset(q_tile, [c0, c0], [1, HALF_D])
                    q_out2 = pto.subset(q_tile, [c0, cHalf], [1, HALF_D])
                    k_out1 = pto.subset(k_tile, [c0, c0], [1, HALF_D])
                    k_out2 = pto.subset(k_tile, [c0, cHalf], [1, HALF_D])

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

                    pto.mov(v_raw, v_tile)

                    pto.store(q_tile, q_view)
                    pto.store(k_tile, k_view)
                    pto.store(v_tile, v_view)

    return dequant_rope_quant_kvcache_rotary_stage


def _build_cache_stage(*, output_dir):
    total_rows = _total_rows()
    block_dim = _block_dim()

    @jit(
        meta_data=_cache_meta_data,
        output_dir=output_dir,
        block_dim=block_dim,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def dequant_rope_quant_kvcache_cache_stage(
        k_ptr: "ptr_fp16",
        v_ptr: "ptr_fp16",
        k_cache_ptr: "ptr_i8",
        v_cache_ptr: "ptr_i8",
        rows_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c8 = const(8)
        cRows = const(total_rows)
        c64 = const(D)

        rows = pto.index_cast(rows_i32)
        k_tensor = pto.as_tensor(tensor_row, ptr=k_ptr, shape=[cRows, c64], strides=[c64, c1])
        v_tensor = pto.as_tensor(tensor_row, ptr=v_ptr, shape=[cRows, c64], strides=[c64, c1])
        k_cache_tensor = pto.as_tensor(tensor_cache, ptr=k_cache_ptr, shape=[cRows * c8, c64], strides=[c64, c1])
        v_cache_tensor = pto.as_tensor(tensor_cache, ptr=v_cache_ptr, shape=[cRows * c8, c64], strides=[c64, c1])

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cRows, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cRows)

            k_tile = pto.alloc_tile(tile_full)
            v_tile = pto.alloc_tile(tile_full)
            k_i8_tile = pto.alloc_tile(tile_full_i8)
            v_i8_tile = pto.alloc_tile(tile_full_i8)
            if rows == cRows:
                for row_idx in range(row_start, row_end, c1):
                    k_view = pto.slice_view(row_view, source=k_tensor, offsets=[row_idx, c0], sizes=[c1, c64])
                    v_view = pto.slice_view(row_view, source=v_tensor, offsets=[row_idx, c0], sizes=[c1, c64])
                    cache_row = row_idx * c8 + row_idx
                    k_cache_view = pto.slice_view(cache_view, source=k_cache_tensor, offsets=[cache_row, c0], sizes=[c1, c64])
                    v_cache_view = pto.slice_view(cache_view, source=v_cache_tensor, offsets=[cache_row, c0], sizes=[c1, c64])

                    pto.load(k_view, k_tile)
                    pto.load(v_view, v_tile)
                    pto.cvt(k_tile, k_i8_tile)
                    pto.cvt(v_tile, v_i8_tile)
                    pto.store(k_i8_tile, k_cache_view)
                    pto.store(v_i8_tile, v_cache_view)

    return dequant_rope_quant_kvcache_cache_stage


class DequantRopeQuantKvcacheWrapper:
    def __init__(self, *, output_dir):
        self._output_dir = Path(output_dir)
        self._rotary = _build_rotary_stage(output_dir=self._output_dir / "stage_rotary")
        self._cache = _build_cache_stage(output_dir=self._output_dir / "stage_cache")

    def _build(self):
        for stage in (self._rotary, self._cache):
            build = getattr(stage, "_build", None)
            if callable(build):
                build()

    def _artifact_paths(self):
        paths = []
        for stage in (self._rotary, self._cache):
            artifact_paths = getattr(stage, "_artifact_paths", None)
            if callable(artifact_paths):
                paths.extend(artifact_paths())
        return tuple(paths)

    @property
    def library_path(self):
        return getattr(self._cache, "library_path", None)

    @property
    def output_dir(self):
        return str(self._output_dir)

    def __call__(self, x, weight_scale, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, stream_ptr=None):
        del scale_k, scale_v
        rows = int(x.shape[0])
        expected = torch.arange(rows, device=indices.device, dtype=indices.dtype)
        if not torch.equal(indices, expected):
            raise NotImplementedError("Current PTODSL dequant_rope_quant_kvcache seed requires indices[row] == row.")
        q_out = torch.empty((rows, 1, D), dtype=cos.dtype, device=x.device)
        k_out = torch.empty((rows, 1, D), dtype=cos.dtype, device=x.device)
        v_out = torch.empty((rows, 1, D), dtype=cos.dtype, device=x.device)
        k_cache_out = k_cache.clone()
        v_cache_out = v_cache.clone()
        self._rotary(
            q_out.view(rows, D),
            k_out.view(rows, D),
            v_out.view(rows, D),
            x.contiguous(),
            weight_scale.view(1, TOTAL_D).contiguous(),
            cos.contiguous(),
            sin.contiguous(),
            rows,
            stream_ptr=stream_ptr,
        )
        self._cache(
            k_out.view(rows, D),
            v_out.view(rows, D),
            k_cache_out,
            v_cache_out,
            rows,
            stream_ptr=stream_ptr,
        )
        return q_out, k_out, v_out, k_cache_out, v_cache_out


def build_jit_wrapper(*, output_dir):
    return DequantRopeQuantKvcacheWrapper(output_dir=output_dir)
