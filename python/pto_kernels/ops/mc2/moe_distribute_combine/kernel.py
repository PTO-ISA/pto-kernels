"""Constrained PTO-DSL seed for moe_distribute_combine on 910B.

This slice covers the local integration stage after the reverse-route buffer
has already been compacted into a per-core local reverse-route buffer by the
host harness. It keeps the real A2 shape contract (`epWorldSize=8`, `H=7168`)
and mirrors the upstream contiguous per-core row ownership, while leaving the distributed
AllToAllV return path as an explicit blocker.
"""

from dataclasses import dataclass

from ptodsl import jit, pto, tile
from ptodsl import scalar as s

from pto_kernels.utils.tuning import tuned_int


const = s.const


@dataclass(frozen=True)
class MoeDistributeCombineConfig:
    tokens: int
    hidden: int
    world_size: int
    block_dim: int

    @property
    def rows_per_core(self) -> int:
        return self.tokens // self.block_dim


def _config() -> MoeDistributeCombineConfig:
    return MoeDistributeCombineConfig(
        tokens=tuned_int("PTO_MC2_MOE_COMBINE_TOKENS", 8, valid_values=(8,)),
        hidden=tuned_int("PTO_MC2_MOE_COMBINE_HIDDEN", 7168, valid_values=(7168,)),
        world_size=tuned_int("PTO_MC2_MOE_COMBINE_WORLD_SIZE", 8, valid_values=(8,)),
        block_dim=tuned_int("PTO_MC2_MOE_COMBINE_BLOCK_DIM", 4, valid_values=(2, 4, 8)),
    )


def _meta_data(config: MoeDistributeCombineConfig):
    dtype = pto.float16
    idx_dtype = pto.int16
    ptr = pto.PtrType(dtype)
    ptr_idx = pto.PtrType(idx_dtype)
    tensor = pto.TensorType(rank=1, dtype=dtype)
    tensor_idx = pto.TensorType(rank=1, dtype=idx_dtype)
    row_view = pto.SubTensorType(shape=[1, config.hidden], dtype=dtype)
    row_idx_view = pto.SubTensorType(shape=[1, config.hidden], dtype=idx_dtype)
    chunk_view = pto.SubTensorType(shape=[1, config.rows_per_core * config.hidden], dtype=dtype)
    row_tile = pto.TileBufType(
        shape=[1, config.hidden],
        valid_shape=[1, config.hidden],
        dtype=dtype,
        memory_space="VEC",
        config=pto.TileBufConfig(),
    )
    row_idx_tile = pto.TileBufType(
        shape=[1, config.hidden],
        valid_shape=[1, config.hidden],
        dtype=idx_dtype,
        memory_space="VEC",
        config=pto.TileBufConfig(),
    )
    chunk_tile = pto.TileBufType(
        shape=[1, config.rows_per_core * config.hidden],
        valid_shape=[1, config.rows_per_core * config.hidden],
        dtype=dtype,
        memory_space="VEC",
        config=pto.TileBufConfig(),
    )
    return {
        "ptr": ptr,
        "ptr_idx": ptr_idx,
        "tensor": tensor,
        "tensor_idx": tensor_idx,
        "row_view": row_view,
        "row_idx_view": row_idx_view,
        "chunk_view": chunk_view,
        "row_tile": row_tile,
        "row_idx_tile": row_idx_tile,
        "chunk_tile": chunk_tile,
    }


def build_jit_wrapper(*, output_dir):
    config = _config()

    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.tokens, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_distribute_combine_seed(x_out_ptr: "ptr", compact_expand_x_ptr: "ptr", scatter_idx_ptr: "ptr_idx") -> None:
        c0 = const(0)
        c1 = const(1)
        cTokens = const(config.tokens)
        cHidden = const(config.hidden)
        cRowsPerCore = const(config.rows_per_core)
        cChunkWidth = const(config.rows_per_core * config.hidden)
        cTotal = const(config.tokens * config.hidden)

        tv_src = pto.as_tensor(
            tensor,
            ptr=compact_expand_x_ptr,
            shape=[cTotal],
            strides=[c1],
        )
        tv_idx = pto.as_tensor(
            tensor_idx,
            ptr=scatter_idx_ptr,
            shape=[cTotal],
            strides=[c1],
        )
        tv_dst = pto.as_tensor(
            tensor,
            ptr=x_out_ptr,
            shape=[cTotal],
            strides=[c1],
        )

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            row_start = bid * cRowsPerCore
            row_end = row_start + cRowsPerCore
            chunk_offset = row_start * cHidden

            src_row_tile = pto.alloc_tile(row_tile)
            idx_row_tile = pto.alloc_tile(row_idx_tile)
            dst_chunk_tile = pto.alloc_tile(chunk_tile)
            dst_chunk_view = pto.slice_view(
                chunk_view,
                source=tv_dst,
                offsets=[chunk_offset],
                sizes=[cChunkWidth],
            )
            pto.load(dst_chunk_view, dst_chunk_tile)

            for row_idx in pto.range(row_start, row_end, c1):
                row_offset = row_idx * cHidden
                src_view = pto.slice_view(row_view, source=tv_src, offsets=[row_offset], sizes=[cHidden])
                idx_view = pto.slice_view(row_idx_view, source=tv_idx, offsets=[row_offset], sizes=[cHidden])
                pto.load(src_view, src_row_tile)
                pto.load(idx_view, idx_row_tile)
                tile.scatter(src_row_tile, idx_row_tile, dst_chunk_tile)
            pto.store(dst_chunk_tile, dst_chunk_view)

    return moe_distribute_combine_seed
