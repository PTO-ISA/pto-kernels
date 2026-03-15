"""Constrained PTO-DSL seed for moe_init_routing on 910B.

This phase-2 slice matches the host contract for top-1 routing with 2D row_idx /
expert_idx inputs, but constrains expert_idx to arrive pre-grouped by expert so
the PTO kernel only needs gather/copy movement. The missing on-device sort
surface remains tracked separately.
"""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class MoeInitRoutingConfig:
    tokens: int
    hidden: int
    block_dim: int

    @property
    def total(self) -> int:
        return self.tokens * self.hidden


def _config() -> MoeInitRoutingConfig:
    return MoeInitRoutingConfig(
        tokens=tuned_int("PTO_MOE_INIT_ROUTING_TOKENS", 16, valid_values=(16, 128, 256)),
        hidden=tuned_int("PTO_MOE_INIT_ROUTING_HIDDEN", 16, valid_values=(16, 64, 128)),
        block_dim=tuned_int("PTO_MOE_INIT_ROUTING_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)),
    )


def _gather_meta_data(config: MoeInitRoutingConfig):
    dtype = pto.float16
    i32 = pto.int32

    ptr = pto.PtrType(dtype)
    ptr_i32 = pto.PtrType(i32)

    tensor = pto.TensorType(rank=1, dtype=dtype)
    tensor_i32 = pto.TensorType(rank=1, dtype=i32)

    sub_tokens = pto.SubTensorType(shape=[1, config.total], dtype=dtype)
    sub_out = pto.SubTensorType(shape=[1, config.hidden], dtype=dtype)
    sub_gather = pto.SubTensorType(shape=[1, config.hidden], dtype=i32)

    cfg = pto.TileConfig()
    tile_tokens = pto.TileType(
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
    tile_out = pto.TileType(
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
        "sub_tokens": sub_tokens,
        "sub_out": sub_out,
        "sub_gather": sub_gather,
        "tile_tokens": tile_tokens,
        "tile_gather": tile_gather,
        "tile_out": tile_out,
    }


def _copy_indices_meta_data(config: MoeInitRoutingConfig):
    i32 = pto.int32
    ptr_i32 = pto.PtrType(i32)
    tensor_i32 = pto.TensorType(rank=1, dtype=i32)
    sub_i32 = pto.SubTensorType(shape=[1, config.tokens], dtype=i32)
    cfg = pto.TileConfig()
    tile_i32 = pto.TileType(
        shape=[1, config.tokens],
        valid_shape=[1, config.tokens],
        dtype=i32,
        memory_space="VEC",
        config=cfg,
    )
    return {
        "ptr_i32": ptr_i32,
        "tensor_i32": tensor_i32,
        "sub_i32": sub_i32,
        "tile_i32": tile_i32,
    }


def _build_gather_kernel(*, config: MoeInitRoutingConfig, output_dir):
    @jit(
        meta_data=lambda: _gather_meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.tokens, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_init_routing_seed(out_ptr: "ptr", x_ptr: "ptr", gather_ptr: "ptr_i32") -> None:
        c0 = const(0)
        c1 = const(1)
        cTokens = const(config.tokens)
        cHidden = const(config.hidden)
        cTotal = const(config.total)

        tv_x = pto.as_tensor(
            tensor,
            ptr=x_ptr,
            shape=[cTotal],
            strides=[c1],
        )
        tv_gather = pto.as_tensor(
            tensor_i32,
            ptr=gather_ptr,
            shape=[cTotal],
            strides=[c1],
        )
        tv_out = pto.as_tensor(
            tensor,
            ptr=out_ptr,
            shape=[cTotal],
            strides=[c1],
        )

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cTokens, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cTokens)
            x_tile = pto.alloc_tile(tile_tokens)
            gather_tile = pto.alloc_tile(tile_gather)
            out_tile = pto.alloc_tile(tile_out)

            x_view = pto.slice_view(sub_tokens, source=tv_x, offsets=[c0], sizes=[cTotal])
            pto.load(x_view, x_tile)

            for row_idx in range(row_start, row_end, c1):
                row_off = row_idx * cHidden
                gather_view = pto.slice_view(sub_gather, source=tv_gather, offsets=[row_off], sizes=[cHidden])
                out_view = pto.slice_view(sub_out, source=tv_out, offsets=[row_off], sizes=[cHidden])
                pto.load(gather_view, gather_tile)
                pto.gather(x_tile, out_tile, gather_tile)
                pto.store(out_tile, out_view)

    return moe_init_routing_seed


def _build_copy_kernel(*, config: MoeInitRoutingConfig, output_dir):
    @jit(
        meta_data=lambda: _copy_indices_meta_data(config),
        output_dir=output_dir,
        block_dim=1,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_init_routing_copy_indices(out_ptr: "ptr_i32", src_ptr: "ptr_i32") -> None:
        c0 = const(0)
        c1 = const(1)
        cTokens = const(config.tokens)

        tv_in = pto.as_tensor(
            tensor_i32,
            ptr=src_ptr,
            shape=[cTokens],
            strides=[c1],
        )
        tv_out = pto.as_tensor(
            tensor_i32,
            ptr=out_ptr,
            shape=[cTokens],
            strides=[c1],
        )

        with pto.section.vector():
            in_tile = pto.alloc_tile(tile_i32)
            in_view = pto.slice_view(sub_i32, source=tv_in, offsets=[c0], sizes=[cTokens])
            out_view = pto.slice_view(sub_i32, source=tv_out, offsets=[c0], sizes=[cTokens])
            pto.load(in_view, in_tile)
            pto.store(in_tile, out_view)

    return moe_init_routing_copy_indices


class MoeInitRoutingWrapper:
    def __init__(self, *, output_dir):
        self._output_dir = Path(output_dir)
        self._config = _config()
        self._gather = _build_gather_kernel(
            config=self._config,
            output_dir=self._output_dir / "stage1_gather",
        )
        self._copy_rows = _build_copy_kernel(
            config=self._config,
            output_dir=self._output_dir / "stage2_copy_rows",
        )
        self._copy_experts = _build_copy_kernel(
            config=self._config,
            output_dir=self._output_dir / "stage3_copy_experts",
        )

    def _build(self):
        self._gather._build()
        self._copy_rows._build()
        self._copy_experts._build()

    def _artifact_paths(self):
        return (
            *self._gather._artifact_paths(),
            *self._copy_rows._artifact_paths(),
            *self._copy_experts._artifact_paths(),
        )

    @property
    def library_path(self):
        return None

    @property
    def output_dir(self):
        return str(self._output_dir)

    def __call__(
        self,
        expanded_x_ptr,
        expanded_row_idx_ptr,
        expanded_expert_idx_ptr,
        x_ptr,
        row_idx_ptr,
        expert_idx_ptr,
        gather_ptr,
        stream_ptr=None,
    ):
        self._gather(expanded_x_ptr, x_ptr, gather_ptr, stream_ptr=stream_ptr)
        self._copy_rows(expanded_row_idx_ptr, row_idx_ptr, stream_ptr=stream_ptr)
        self._copy_experts(expanded_expert_idx_ptr, expert_idx_ptr, stream_ptr=stream_ptr)


def build_jit_wrapper(*, output_dir):
    return MoeInitRoutingWrapper(output_dir=output_dir)
