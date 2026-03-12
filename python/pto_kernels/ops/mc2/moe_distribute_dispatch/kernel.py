"""Constrained PTO-DSL seed for moe_distribute_dispatch on 910B.

The first slice mirrors the local pack/reorder phase of the upstream MC2
dispatch path for an EP-only, top-1, no-quant contract. PTO owns the local
destination-major send-buffer pack; the benchmark harness still owns the HCCL
all-to-all and the metadata tensors needed by the paired combine path.
"""

from pathlib import Path

from ptodsl import jit, pto

from pto_kernels.utils.tuning import tuned_int


const = pto.const


class MoeDistributeDispatchConfig:
    def __init__(self, *, tokens: int, hidden: int, world_size: int, block_dim: int):
        self.tokens = tokens
        self.hidden = hidden
        self.world_size = world_size
        self.block_dim = block_dim

    @property
    def total(self) -> int:
        return self.tokens * self.hidden


def _config() -> MoeDistributeDispatchConfig:
    return MoeDistributeDispatchConfig(
        tokens=tuned_int("PTO_MC2_MOE_DISPATCH_TOKENS", 8, valid_values=(8, 16, 32)),
        hidden=tuned_int("PTO_MC2_MOE_DISPATCH_HIDDEN", 7168, valid_values=(7168,)),
        world_size=tuned_int("PTO_MC2_MOE_DISPATCH_WORLD_SIZE", 8, valid_values=(8,)),
        block_dim=tuned_int("PTO_MC2_MOE_DISPATCH_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)),
    )


def _meta_data(config: MoeDistributeDispatchConfig):
    dtype = pto.float16
    i32 = pto.int32

    ptr = pto.PtrType(dtype)
    ptr_i32 = pto.PtrType(i32)
    tensor = pto.TensorType(rank=1, dtype=dtype)
    tensor_i32 = pto.TensorType(rank=1, dtype=i32)

    sub_src = pto.SubTensorType(shape=[1, config.total], dtype=dtype)
    sub_gather = pto.SubTensorType(shape=[1, config.hidden], dtype=i32)
    sub_dst = pto.SubTensorType(shape=[1, config.hidden], dtype=dtype)

    cfg = pto.TileConfig()
    tile_src = pto.TileType(
        shape=[1, config.total],
        valid_shape=[1, config.total],
        dtype=dtype,
        memory_space="VEC",
        config=cfg,
    )
    tile_gather = pto.TileType(
        shape=[1, config.hidden],
        valid_shape=[1, config.hidden],
        dtype=i32,
        memory_space="VEC",
        config=cfg,
    )
    tile_dst = pto.TileType(
        shape=[1, config.hidden],
        valid_shape=[1, config.hidden],
        dtype=dtype,
        memory_space="VEC",
        config=cfg,
    )

    return {
        "ptr": ptr,
        "ptr_i32": ptr_i32,
        "tensor": tensor,
        "tensor_i32": tensor_i32,
        "sub_src": sub_src,
        "sub_gather": sub_gather,
        "sub_dst": sub_dst,
        "tile_src": tile_src,
        "tile_gather": tile_gather,
        "tile_dst": tile_dst,
    }


def _build_pack_kernel(*, config: MoeDistributeDispatchConfig, output_dir):
    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.tokens, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_distribute_dispatch_pack(send_ptr: "ptr", x_ptr: "ptr", gather_ptr: "ptr_i32") -> None:
        c0 = const(0)
        c1 = const(1)
        cTokens = const(config.tokens)
        cHidden = const(config.hidden)
        cTotal = const(config.total)

        tv_src = pto.as_tensor(
            tensor,
            ptr=x_ptr,
            shape=[cTotal],
            strides=[c1],
        )
        tv_row_idx = pto.as_tensor(
            tensor_i32,
            ptr=gather_ptr,
            shape=[cTotal],
            strides=[c1],
        )
        tv_dst = pto.as_tensor(
            tensor,
            ptr=send_ptr,
            shape=[cTotal],
            strides=[c1],
        )

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cTokens, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cTokens)

            src_tile = pto.alloc_tile(tile_src)
            gather_tile = pto.alloc_tile(tile_gather)
            dst_tile = pto.alloc_tile(tile_dst)

            src_view = pto.slice_view(sub_src, source=tv_src, offsets=[c0], sizes=[cTotal])
            pto.load(src_view, src_tile)

            for row_idx in range(row_start, row_end, c1):
                row_off = row_idx * cHidden
                gather_view = pto.slice_view(sub_gather, source=tv_row_idx, offsets=[row_off], sizes=[cHidden])
                dst_view = pto.slice_view(sub_dst, source=tv_dst, offsets=[row_off], sizes=[cHidden])
                pto.load(gather_view, gather_tile)
                pto.gather(src_tile, dst_tile, indices=gather_tile)
                pto.store(dst_tile, dst_view)

    return moe_distribute_dispatch_pack


class MoeDistributeDispatchWrapper:
    def __init__(self, *, output_dir):
        self._output_dir = Path(output_dir)
        self._config = _config()
        self._pack = _build_pack_kernel(
            config=self._config,
            output_dir=self._output_dir / "pack_send_buffer",
        )

    def _build(self):
        self._pack._build()

    def _artifact_paths(self):
        return tuple(self._pack._artifact_paths())

    @property
    def library_path(self):
        return None

    @property
    def output_dir(self):
        return str(self._output_dir)

    def __call__(self, send_ptr, x_ptr, gather_ptr, stream_ptr=None):
        self._pack(send_ptr, x_ptr, gather_ptr, stream_ptr=stream_ptr)


def build_jit_wrapper(*, output_dir):
    return MoeDistributeDispatchWrapper(output_dir=output_dir)
