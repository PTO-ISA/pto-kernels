"""Constrained PTO-DSL seed for ring_attention_update on 910B."""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class RingAttentionUpdateConfig:
    rows: int
    head_dim: int
    block_dim: int


def _config() -> RingAttentionUpdateConfig:
    return RingAttentionUpdateConfig(
        rows=tuned_int("PTO_RING_ATTENTION_UPDATE_ROWS", 8, valid_values=(8, 128, 256)),
        head_dim=tuned_int("PTO_RING_ATTENTION_UPDATE_HEAD_DIM", 16, valid_values=(16, 64, 128)),
        block_dim=tuned_int("PTO_RING_ATTENTION_UPDATE_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)),
    )


def _meta_data(config: RingAttentionUpdateConfig):
    fp16 = pto.float16
    fp32 = pto.float32
    ptr_fp16 = pto.PtrType(fp16)
    ptr_fp32 = pto.PtrType(fp32)
    tensor_fp16 = pto.TensorType(rank=2, dtype=fp16)
    tensor_fp32 = pto.TensorType(rank=2, dtype=fp32)
    row_view_fp16 = pto.SubTensorType(shape=[1, config.head_dim], dtype=fp16)
    row8_view_fp32 = pto.SubTensorType(shape=[1, 8], dtype=fp32)
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
    row8_tile_fp32 = pto.TileType(
        shape=[1, 8],
        valid_shape=[1, 8],
        dtype=fp32,
        memory_space="VEC",
        config=pto.TileConfig(),
    )
    row8_scalar_tile_fp32 = pto.TileType(
        shape=[1, 8],
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
        "row8_view_fp32": row8_view_fp32,
        "row_tile_fp16": row_tile_fp16,
        "row_tile_fp32": row_tile_fp32,
        "row8_tile_fp32": row8_tile_fp32,
        "row8_scalar_tile_fp32": row8_scalar_tile_fp32,
    }


def _build_kernel(*, config: RingAttentionUpdateConfig, output_dir):
    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.rows, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def ring_attention_update(
        attn_out_ptr: "ptr_fp16",
        softmax_max_out_ptr: "ptr_fp32",
        softmax_sum_out_ptr: "ptr_fp32",
        prev_attn_out_ptr: "ptr_fp16",
        prev_softmax_max_ptr: "ptr_fp32",
        prev_softmax_sum_ptr: "ptr_fp32",
        cur_attn_out_ptr: "ptr_fp16",
        cur_softmax_max_ptr: "ptr_fp32",
        cur_softmax_sum_ptr: "ptr_fp32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c8 = const(8)
        cRows = const(config.rows)
        cD = const(config.head_dim)
        inv8 = const(1.0 / 8.0, dtype=pto.float32)

        prev_attn_out = pto.as_tensor(tensor_fp16, ptr=prev_attn_out_ptr, shape=[cRows, cD], strides=[cD, c1])
        cur_attn_out = pto.as_tensor(tensor_fp16, ptr=cur_attn_out_ptr, shape=[cRows, cD], strides=[cD, c1])
        attn_out = pto.as_tensor(tensor_fp16, ptr=attn_out_ptr, shape=[cRows, cD], strides=[cD, c1])

        softmax_max_out = pto.as_tensor(
            tensor_fp32,
            ptr=softmax_max_out_ptr,
            shape=[cRows, c8],
            strides=[c8, c1],
        )
        softmax_sum_out = pto.as_tensor(
            tensor_fp32,
            ptr=softmax_sum_out_ptr,
            shape=[cRows, c8],
            strides=[c8, c1],
        )
        prev_softmax_max = pto.as_tensor(
            tensor_fp32,
            ptr=prev_softmax_max_ptr,
            shape=[cRows, c8],
            strides=[c8, c1],
        )
        prev_softmax_sum = pto.as_tensor(
            tensor_fp32,
            ptr=prev_softmax_sum_ptr,
            shape=[cRows, c8],
            strides=[c8, c1],
        )
        cur_softmax_max = pto.as_tensor(
            tensor_fp32,
            ptr=cur_softmax_max_ptr,
            shape=[cRows, c8],
            strides=[c8, c1],
        )
        cur_softmax_sum = pto.as_tensor(
            tensor_fp32,
            ptr=cur_softmax_sum_ptr,
            shape=[cRows, c8],
            strides=[c8, c1],
        )

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cRows, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cRows)

            prev_f16 = pto.alloc_tile(row_tile_fp16, valid_col=cD)
            cur_f16 = pto.alloc_tile(row_tile_fp16, valid_col=cD)
            out_f16 = pto.alloc_tile(row_tile_fp16, valid_col=cD)

            prev_f32 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            cur_f32 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            prev_scale = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            cur_scale = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            prev_sum_row = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            cur_sum_row = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            denom = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            tmp0 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            tmp1 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            prev_factor8 = pto.alloc_tile(row8_tile_fp32)
            cur_factor8 = pto.alloc_tile(row8_tile_fp32)
            prev_sum8 = pto.alloc_tile(row8_tile_fp32)
            cur_sum8 = pto.alloc_tile(row8_tile_fp32)
            sum_tile = pto.alloc_tile(row8_tile_fp32)
            max_tile = pto.alloc_tile(row8_tile_fp32)
            prev_tmp8 = pto.alloc_tile(row8_tile_fp32)
            cur_tmp8 = pto.alloc_tile(row8_tile_fp32)
            merged_tmp8 = pto.alloc_tile(row8_tile_fp32)
            prev_max_scalar = pto.alloc_tile(row8_scalar_tile_fp32, valid_col=c1)
            cur_max_scalar = pto.alloc_tile(row8_scalar_tile_fp32, valid_col=c1)
            merged_max_scalar = pto.alloc_tile(row8_scalar_tile_fp32, valid_col=c1)
            prev_sum_scalar = pto.alloc_tile(row8_scalar_tile_fp32, valid_col=c1)
            cur_sum_scalar = pto.alloc_tile(row8_scalar_tile_fp32, valid_col=c1)
            merged_sum_scalar = pto.alloc_tile(row8_scalar_tile_fp32, valid_col=c1)

            for row_idx in range(row_start, row_end, c1):
                prev_view = pto.slice_view(row_view_fp16, source=prev_attn_out, offsets=[row_idx, c0], sizes=[c1, cD])
                cur_view = pto.slice_view(row_view_fp16, source=cur_attn_out, offsets=[row_idx, c0], sizes=[c1, cD])
                out_view = pto.slice_view(row_view_fp16, source=attn_out, offsets=[row_idx, c0], sizes=[c1, cD])
                max_view = pto.slice_view(row8_view_fp32, source=softmax_max_out, offsets=[row_idx, c0], sizes=[c1, c8])
                sum_view = pto.slice_view(row8_view_fp32, source=softmax_sum_out, offsets=[row_idx, c0], sizes=[c1, c8])
                prev_max_view = pto.slice_view(
                    row8_view_fp32,
                    source=prev_softmax_max,
                    offsets=[row_idx, c0],
                    sizes=[c1, c8],
                )
                prev_sum_view = pto.slice_view(
                    row8_view_fp32,
                    source=prev_softmax_sum,
                    offsets=[row_idx, c0],
                    sizes=[c1, c8],
                )
                cur_max_view = pto.slice_view(
                    row8_view_fp32,
                    source=cur_softmax_max,
                    offsets=[row_idx, c0],
                    sizes=[c1, c8],
                )
                cur_sum_view = pto.slice_view(
                    row8_view_fp32,
                    source=cur_softmax_sum,
                    offsets=[row_idx, c0],
                    sizes=[c1, c8],
                )

                pto.load(prev_view, prev_f16)
                pto.load(cur_view, cur_f16)
                pto.load(prev_max_view, prev_factor8)
                pto.load(prev_sum_view, prev_sum8)
                pto.load(cur_max_view, cur_factor8)
                pto.load(cur_sum_view, cur_sum8)
                pto.cvt(prev_f16, prev_f32)
                pto.cvt(cur_f16, cur_f32)

                pto.row_max(prev_factor8, prev_tmp8, prev_max_scalar)
                pto.row_max(cur_factor8, cur_tmp8, cur_max_scalar)
                pto.max(prev_max_scalar, cur_max_scalar, merged_max_scalar)
                pto.sub(prev_max_scalar, merged_max_scalar, prev_tmp8)
                pto.sub(cur_max_scalar, merged_max_scalar, cur_tmp8)
                pto.row_expand(prev_tmp8, prev_scale)
                pto.row_expand(cur_tmp8, cur_scale)
                pto.exp(prev_scale, prev_scale)
                pto.exp(cur_scale, cur_scale)
                pto.row_sum(prev_sum8, prev_tmp8, prev_sum_scalar)
                pto.row_sum(cur_sum8, cur_tmp8, cur_sum_scalar)
                pto.muls(prev_sum_scalar, inv8, prev_sum_scalar)
                pto.muls(cur_sum_scalar, inv8, cur_sum_scalar)
                pto.row_expand(prev_sum_scalar, prev_sum_row)
                pto.row_expand(cur_sum_scalar, cur_sum_row)
                pto.mul(prev_scale, prev_sum_row, prev_scale)
                pto.mul(cur_scale, cur_sum_row, cur_scale)
                pto.add(prev_scale, cur_scale, denom)
                pto.div(prev_scale, denom, prev_scale)
                pto.div(cur_scale, denom, cur_scale)

                pto.mul(prev_f32, prev_scale, tmp0)
                pto.mul(cur_f32, cur_scale, tmp1)
                pto.add(tmp0, tmp1, tmp0)
                pto.cvt(tmp0, out_f16)
                pto.store(out_f16, out_view)

                pto.row_expand(merged_max_scalar, max_tile)
                pto.sub(prev_max_scalar, merged_max_scalar, prev_tmp8)
                pto.sub(cur_max_scalar, merged_max_scalar, cur_tmp8)
                pto.row_expand(prev_tmp8, prev_factor8)
                pto.row_expand(cur_tmp8, cur_factor8)
                pto.exp(prev_factor8, prev_factor8)
                pto.exp(cur_factor8, cur_factor8)
                pto.row_expand(prev_sum_scalar, prev_sum8)
                pto.row_expand(cur_sum_scalar, cur_sum8)
                pto.mul(prev_factor8, prev_sum8, prev_factor8)
                pto.mul(cur_factor8, cur_sum8, cur_factor8)
                pto.add(prev_factor8, cur_factor8, sum_tile)
                pto.row_sum(sum_tile, merged_tmp8, merged_sum_scalar)
                pto.muls(merged_sum_scalar, inv8, merged_sum_scalar)
                pto.row_expand(merged_sum_scalar, sum_tile)
                pto.store(max_tile, max_view)
                pto.store(sum_tile, sum_view)

    return ring_attention_update


class RingAttentionUpdateWrapper:
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

    def __call__(
        self,
        attn_out,
        softmax_max_out,
        softmax_sum_out,
        prev_attn_out,
        prev_softmax_max,
        prev_softmax_sum,
        cur_attn_out,
        cur_softmax_max,
        cur_softmax_sum,
        stream_ptr=None,
    ):
        self._kernel(
            attn_out,
            softmax_max_out,
            softmax_sum_out,
            prev_attn_out,
            prev_softmax_max,
            prev_softmax_sum,
            cur_attn_out,
            cur_softmax_max,
            cur_softmax_sum,
            stream_ptr=stream_ptr,
        )


def build_jit_wrapper(*, output_dir):
    return RingAttentionUpdateWrapper(output_dir=output_dir)
