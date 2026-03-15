"""Constrained PTO-DSL seed for attention_update on 910B."""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class AttentionUpdateConfig:
    rows: int
    head_dim: int
    block_dim: int


def _config() -> AttentionUpdateConfig:
    return AttentionUpdateConfig(
        rows=tuned_int("PTO_ATTENTION_UPDATE_ROWS", 8, valid_values=(8, 128, 256)),
        head_dim=tuned_int("PTO_ATTENTION_UPDATE_HEAD_DIM", 16, valid_values=(16, 64, 128)),
        block_dim=tuned_int("PTO_ATTENTION_UPDATE_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)),
    )


def _meta_data(config: AttentionUpdateConfig):
    fp16 = pto.float16
    fp32 = pto.float32
    ptr_fp16 = pto.PtrType(fp16)
    ptr_fp32 = pto.PtrType(fp32)
    tensor_fp16 = pto.TensorType(rank=2, dtype=fp16)
    tensor_fp32 = pto.TensorType(rank=2, dtype=fp32)
    row_view_fp16 = pto.SubTensorType(shape=[1, config.head_dim], dtype=fp16)
    scalar_view_fp32 = pto.SubTensorType(shape=[1, 1], dtype=fp32)
    row_tile_fp16 = pto.TileType(
        shape=[1, config.head_dim],
        valid_shape=[1, -1],
        dtype=fp16,
        memory_space="VEC",
        config=pto.TileConfig(),
    )
    row_tile_fp32 = pto.TileType(
        shape=[1, config.head_dim],
        valid_shape=[1, -1],
        dtype=fp32,
        memory_space="VEC",
        config=pto.TileConfig(),
    )
    return {
        "ptr_fp16": ptr_fp16,
        "ptr_fp32": ptr_fp32,
        "tensor_fp16": tensor_fp16,
        "tensor_fp32": tensor_fp32,
        "row_view_fp16": row_view_fp16,
        "scalar_view_fp32": scalar_view_fp32,
        "row_tile_fp16": row_tile_fp16,
        "row_tile_fp32": row_tile_fp32,
    }


def _build_kernel(*, config: AttentionUpdateConfig, output_dir):
    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.rows, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def attention_update(
        out_ptr: "ptr_fp16",
        lse0_ptr: "ptr_fp32",
        lse1_ptr: "ptr_fp32",
        local_out0_ptr: "ptr_fp16",
        local_out1_ptr: "ptr_fp16",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cRows = const(config.rows)
        cD = const(config.head_dim)

        local_out0 = pto.as_tensor(tensor_fp16, ptr=local_out0_ptr, shape=[cRows, cD], strides=[cD, c1])
        local_out1 = pto.as_tensor(tensor_fp16, ptr=local_out1_ptr, shape=[cRows, cD], strides=[cD, c1])
        out = pto.as_tensor(tensor_fp16, ptr=out_ptr, shape=[cRows, cD], strides=[cD, c1])
        lse0 = pto.as_tensor(tensor_fp32, ptr=lse0_ptr, shape=[cRows, c1], strides=[c1, c1])
        lse1 = pto.as_tensor(tensor_fp32, ptr=lse1_ptr, shape=[cRows, c1], strides=[c1, c1])

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cRows, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cRows)

            local0_f16 = pto.alloc_tile(row_tile_fp16, valid_col=cD)
            local1_f16 = pto.alloc_tile(row_tile_fp16, valid_col=cD)
            out_f16 = pto.alloc_tile(row_tile_fp16, valid_col=cD)

            local0_f32 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            local1_f32 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            weight0 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            weight1 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            denom = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            tmp0 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            tmp1 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            lse0_scalar = pto.alloc_tile(row_tile_fp32, valid_col=c1)
            lse1_scalar = pto.alloc_tile(row_tile_fp32, valid_col=c1)
            lse_max_scalar = pto.alloc_tile(row_tile_fp32, valid_col=c1)
            lse0_shift = pto.alloc_tile(row_tile_fp32, valid_col=c1)
            lse1_shift = pto.alloc_tile(row_tile_fp32, valid_col=c1)

            for row_idx in range(row_start, row_end, c1):
                local0_view = pto.slice_view(row_view_fp16, source=local_out0, offsets=[row_idx, c0], sizes=[c1, cD])
                local1_view = pto.slice_view(row_view_fp16, source=local_out1, offsets=[row_idx, c0], sizes=[c1, cD])
                out_view = pto.slice_view(row_view_fp16, source=out, offsets=[row_idx, c0], sizes=[c1, cD])
                lse0_view = pto.slice_view(scalar_view_fp32, source=lse0, offsets=[row_idx, c0], sizes=[c1, c1])
                lse1_view = pto.slice_view(scalar_view_fp32, source=lse1, offsets=[row_idx, c0], sizes=[c1, c1])

                pto.load(local0_view, local0_f16)
                pto.load(local1_view, local1_f16)
                pto.load(lse0_view, lse0_scalar)
                pto.load(lse1_view, lse1_scalar)
                pto.cvt(local0_f16, local0_f32)
                pto.cvt(local1_f16, local1_f32)

                pto.max(lse0_scalar, lse1_scalar, lse_max_scalar)
                pto.sub(lse0_scalar, lse_max_scalar, lse0_shift)
                pto.sub(lse1_scalar, lse_max_scalar, lse1_shift)
                pto.row_expand(lse0_shift, weight0)
                pto.row_expand(lse1_shift, weight1)
                pto.exp(weight0, weight0)
                pto.exp(weight1, weight1)
                pto.add(weight0, weight1, denom)
                pto.div(weight0, denom, weight0)
                pto.div(weight1, denom, weight1)

                pto.mul(local0_f32, weight0, tmp0)
                pto.mul(local1_f32, weight1, tmp1)
                pto.add(tmp0, tmp1, tmp0)
                pto.cvt(tmp0, out_f16)
                pto.store(out_f16, out_view)

    return attention_update


class AttentionUpdateWrapper:
    def __init__(self, *, output_dir):
        self._output_dir = Path(output_dir)
        self._config = _config()
        self._kernel = _build_kernel(config=self._config, output_dir=self._output_dir)

    def _build(self):
        self._kernel._build()

    def _artifact_paths(self):
        return tuple(self._kernel._artifact_paths())

    @property
    def output_dir(self):
        return str(self._output_dir)

    def __call__(self, out, lse0, lse1, local_out0, local_out1, stream_ptr=None):
        self._kernel(out, lse0, lse1, local_out0, local_out1, stream_ptr=stream_ptr)


def build_jit_wrapper(*, output_dir):
    return AttentionUpdateWrapper(output_dir=output_dir)
