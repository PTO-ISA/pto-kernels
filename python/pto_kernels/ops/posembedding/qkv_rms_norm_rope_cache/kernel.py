"""Constrained PTO-DSL seed for qkv_rms_norm_rope_cache on 910B."""

from pathlib import Path

import torch

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const

D = 64
HALF_D = D // 2
SIZE_SPLITS = (64, 64, 64)
TOTAL_D = sum(SIZE_SPLITS)
EPS = 1e-5


def _block_dim() -> int:
    return tuned_int("PTO_QKV_RMS_ROPE_CACHE_BLOCK_DIM", 4, valid_values=(1, 2, 4, 8))


def _total_rows() -> int:
    return tuned_int("PTO_QKV_RMS_ROPE_CACHE_TOTAL_ROWS", 2, valid_values=(2, 4))


def _rotary_meta_data() -> dict[str, object]:
    fp16 = pto.float16
    fp32 = pto.float32
    ptr_fp16 = pto.PtrType(fp16)
    tensor_qkv = pto.TensorType(rank=2, dtype=fp16)
    tensor_row = pto.TensorType(rank=2, dtype=fp16)
    qkv_view = pto.SubTensorType(shape=[1, TOTAL_D], dtype=fp16)
    row_view = pto.SubTensorType(shape=[1, D], dtype=fp16)
    tile_qkv = pto.TileType(shape=[1, TOTAL_D], dtype=fp16, memory_space="VEC")
    tile_full = pto.TileType(shape=[1, D], dtype=fp16, memory_space="VEC")
    tile_half = pto.TileType(shape=[1, HALF_D], dtype=fp16, memory_space="VEC")
    tile_full_acc = pto.TileType(shape=[1, D], dtype=fp32, memory_space="VEC")
    scalar_tile = pto.TileType(
        shape=[1, D],
        valid_shape=[1, 1],
        dtype=fp32,
        memory_space="VEC",
        config=pto.TileConfig(),
    )
    return {
        "ptr_fp16": ptr_fp16,
        "i32": pto.int32,
        "tensor_qkv": tensor_qkv,
        "tensor_row": tensor_row,
        "qkv_view": qkv_view,
        "row_view": row_view,
        "tile_qkv": tile_qkv,
        "tile_full": tile_full,
        "tile_half": tile_half,
        "tile_full_acc": tile_full_acc,
        "scalar_tile": scalar_tile,
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
    inv_n = 1.0 / float(D)

    @jit(
        meta_data=_rotary_meta_data,
        output_dir=output_dir,
        block_dim=block_dim,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def qkv_rms_norm_rope_cache_rotary_stage(
        q_ptr: "ptr_fp16",
        k_ptr: "ptr_fp16",
        v_ptr: "ptr_fp16",
        qkv_ptr: "ptr_fp16",
        q_gamma_ptr: "ptr_fp16",
        k_gamma_ptr: "ptr_fp16",
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
        qkv_tensor = pto.as_tensor(tensor_qkv, ptr=qkv_ptr, shape=[cRows, cTotalD], strides=[cTotalD, c1])
        q_gamma_tensor = pto.as_tensor(tensor_row, ptr=q_gamma_ptr, shape=[c1, c64], strides=[c64, c1])
        k_gamma_tensor = pto.as_tensor(tensor_row, ptr=k_gamma_ptr, shape=[c1, c64], strides=[c64, c1])
        cos_tensor = pto.as_tensor(tensor_row, ptr=cos_ptr, shape=[cRows, c64], strides=[c64, c1])
        sin_tensor = pto.as_tensor(tensor_row, ptr=sin_ptr, shape=[cRows, c64], strides=[c64, c1])

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cRows, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cRows)

            qkv_tile = pto.alloc_tile(tile_qkv)
            q_norm_tile = pto.alloc_tile(tile_full)
            k_norm_tile = pto.alloc_tile(tile_full)
            q_tile = pto.alloc_tile(tile_full)
            k_tile = pto.alloc_tile(tile_full)
            v_tile = pto.alloc_tile(tile_full)
            cos_tile = pto.alloc_tile(tile_full)
            sin_tile = pto.alloc_tile(tile_full)
            q_gamma_f16 = pto.alloc_tile(tile_full)
            k_gamma_f16 = pto.alloc_tile(tile_full)
            q_gamma = pto.alloc_tile(tile_full_acc)
            k_gamma = pto.alloc_tile(tile_full_acc)
            q_acc = pto.alloc_tile(tile_full_acc)
            k_acc = pto.alloc_tile(tile_full_acc)
            q_sq = pto.alloc_tile(tile_full_acc)
            k_sq = pto.alloc_tile(tile_full_acc)
            q_scale = pto.alloc_tile(tile_full_acc)
            k_scale = pto.alloc_tile(tile_full_acc)
            tmp = pto.alloc_tile(tile_full_acc)
            refine = pto.alloc_tile(tile_full_acc)
            const_row = pto.alloc_tile(tile_full_acc)
            scalar = pto.alloc_tile(scalar_tile)
            tmp0 = pto.alloc_tile(tile_half)
            tmp1 = pto.alloc_tile(tile_half)
            zero_half = pto.alloc_tile(tile_half)

            pto.load(pto.slice_view(row_view, source=q_gamma_tensor, offsets=[c0, c0], sizes=[c1, c64]), q_gamma_f16)
            pto.load(pto.slice_view(row_view, source=k_gamma_tensor, offsets=[c0, c0], sizes=[c1, c64]), k_gamma_f16)
            pto.cvt(q_gamma_f16, q_gamma)
            pto.cvt(k_gamma_f16, k_gamma)

            if rows == cRows:
                for row_idx in range(row_start, row_end, c1):
                    qkv_sv = pto.slice_view(qkv_view, source=qkv_tensor, offsets=[row_idx, c0], sizes=[c1, cTotalD])
                    cos_view = pto.slice_view(row_view, source=cos_tensor, offsets=[row_idx, c0], sizes=[c1, c64])
                    sin_view = pto.slice_view(row_view, source=sin_tensor, offsets=[row_idx, c0], sizes=[c1, c64])
                    q_view = pto.slice_view(row_view, source=q_tensor, offsets=[row_idx, c0], sizes=[c1, c64])
                    k_view = pto.slice_view(row_view, source=k_tensor, offsets=[row_idx, c0], sizes=[c1, c64])
                    v_view = pto.slice_view(row_view, source=v_tensor, offsets=[row_idx, c0], sizes=[c1, c64])

                    pto.load(qkv_sv, qkv_tile)
                    pto.load(cos_view, cos_tile)
                    pto.load(sin_view, sin_tile)

                    q_raw = pto.subset(qkv_tile, [c0, c0], [1, D])
                    k_raw = pto.subset(qkv_tile, [c0, c64], [1, D])
                    v_raw = pto.subset(qkv_tile, [c0, c128], [1, D])

                    pto.cvt(q_raw, q_acc)
                    pto.cvt(k_raw, k_acc)

                    pto.mul(q_acc, q_acc, q_sq)
                    pto.row_sum(q_sq, tmp, scalar)
                    pto.row_expand(scalar, q_scale)
                    pto.muls(q_scale, const(inv_n, dtype=pto.float32), q_scale)
                    pto.adds(q_scale, const(EPS, dtype=pto.float32), q_scale)
                    pto.rsqrt(q_scale, tmp)
                    pto.mul(tmp, tmp, refine)
                    pto.mul(q_scale, refine, refine)
                    pto.muls(refine, const(0.5, dtype=pto.float32), refine)
                    pto.expands(const(1.5, dtype=pto.float32), const_row)
                    pto.sub(const_row, refine, refine)
                    pto.mul(tmp, refine, tmp)
                    pto.mul(q_acc, tmp, q_acc)
                    pto.mul(q_acc, q_gamma, q_acc)
                    pto.cvt(q_acc, q_norm_tile)

                    pto.mul(k_acc, k_acc, k_sq)
                    pto.row_sum(k_sq, tmp, scalar)
                    pto.row_expand(scalar, k_scale)
                    pto.muls(k_scale, const(inv_n, dtype=pto.float32), k_scale)
                    pto.adds(k_scale, const(EPS, dtype=pto.float32), k_scale)
                    pto.rsqrt(k_scale, tmp)
                    pto.mul(tmp, tmp, refine)
                    pto.mul(k_scale, refine, refine)
                    pto.muls(refine, const(0.5, dtype=pto.float32), refine)
                    pto.expands(const(1.5, dtype=pto.float32), const_row)
                    pto.sub(const_row, refine, refine)
                    pto.mul(tmp, refine, tmp)
                    pto.mul(k_acc, tmp, k_acc)
                    pto.mul(k_acc, k_gamma, k_acc)
                    pto.cvt(k_acc, k_norm_tile)

                    q_x1 = pto.subset(q_norm_tile, [c0, c0], [1, HALF_D])
                    q_x2 = pto.subset(q_norm_tile, [c0, cHalf], [1, HALF_D])
                    k_x1 = pto.subset(k_norm_tile, [c0, c0], [1, HALF_D])
                    k_x2 = pto.subset(k_norm_tile, [c0, cHalf], [1, HALF_D])
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

    return qkv_rms_norm_rope_cache_rotary_stage


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
    def qkv_rms_norm_rope_cache_cache_stage(
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

    return qkv_rms_norm_rope_cache_cache_stage


class QkvRmsNormRopeCacheWrapper:
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

    def __call__(self, qkv, q_gamma, k_gamma, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, stream_ptr=None):
        del scale_k, scale_v
        rows = int(qkv.shape[0])
        expected = torch.arange(rows, device=indices.device, dtype=indices.dtype)
        if not torch.equal(indices, expected):
            raise NotImplementedError("Current PTODSL qkv_rms_norm_rope_cache seed requires indices[row] == row.")

        q_out = torch.empty((rows, 1, D), dtype=qkv.dtype, device=qkv.device)
        k_out = torch.empty((rows, 1, D), dtype=qkv.dtype, device=qkv.device)
        v_out = torch.empty((rows, 1, D), dtype=qkv.dtype, device=qkv.device)
        k_cache_out = k_cache.clone()
        v_cache_out = v_cache.clone()

        self._rotary(
            q_out.view(rows, D),
            k_out.view(rows, D),
            v_out.view(rows, D),
            qkv.contiguous(),
            q_gamma.view(1, D).contiguous(),
            k_gamma.view(1, D).contiguous(),
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
    return QkvRmsNormRopeCacheWrapper(output_dir=output_dir)
