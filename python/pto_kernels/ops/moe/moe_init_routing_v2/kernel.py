"""Constrained PTO-DSL seed for moe_init_routing_v2 on 910B."""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class MoeInitRoutingV2Config:
    tokens: int
    hidden: int
    experts: int
    block_dim: int


def _config() -> MoeInitRoutingV2Config:
    return MoeInitRoutingV2Config(
        tokens=tuned_int("PTO_MOE_INIT_ROUTING_V2_TOKENS", 16, valid_values=(16, 128, 256)),
        hidden=tuned_int("PTO_MOE_INIT_ROUTING_V2_HIDDEN", 16, valid_values=(16, 64, 128)),
        experts=tuned_int("PTO_MOE_INIT_ROUTING_V2_EXPERTS", 4, valid_values=(4, 8)),
        block_dim=tuned_int("PTO_MOE_INIT_ROUTING_V2_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)),
    )


def _meta_data(config: MoeInitRoutingV2Config):
    dtype = pto.float16
    i32 = pto.int32

    ptr = pto.PtrType(dtype)
    ptr_i32 = pto.PtrType(i32)
    tensor = pto.TensorType(rank=2, dtype=dtype)
    sub_row = pto.SubTensorType(shape=[1, config.hidden], dtype=dtype)

    cfg = pto.TileConfig()
    tile_row = pto.TileType(
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
        "sub_row": sub_row,
        "tile_row": tile_row,
    }


def _build_kernel(*, config: MoeInitRoutingV2Config, output_dir):
    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.tokens, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_init_routing_v2_seed(
        expanded_x_out_ptr: "ptr",
        expanded_row_idx_out_ptr: "ptr_i32",
        expert_tokens_count_or_cumsum_out_ptr: "ptr_i32",
        expert_tokens_before_capacity_out_ptr: "ptr_i32",
        x_ptr: "ptr",
        expert_idx_ptr: "ptr_i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cTokens = const(config.tokens)
        cHidden = const(config.hidden)
        i32 = pto.int32
        c0_i32 = const(0, i32)

        tv_x = pto.as_tensor(
            tensor,
            ptr=x_ptr,
            shape=[cTokens, cHidden],
            strides=[cHidden, c1],
        )
        tv_expanded_x = pto.as_tensor(
            tensor,
            ptr=expanded_x_out_ptr,
            shape=[cTokens, cHidden],
            strides=[cHidden, c1],
        )

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cTokens, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cTokens)
            token_row = pto.alloc_tile(tile_row)

            for row_idx in range(row_start, row_end, c1):
                src_view = pto.slice_view(sub_row, source=tv_x, offsets=[row_idx, c0], sizes=[c1, cHidden])
                dst_view = pto.slice_view(sub_row, source=tv_expanded_x, offsets=[row_idx, c0], sizes=[c1, cHidden])
                pto.load(src_view, token_row)
                pto.store(token_row, dst_view)
                pto.store_scalar(expanded_row_idx_out_ptr, row_idx, pto.index_cast(row_idx, i32))

            if bid == c0:
                for expert in range(config.experts):
                    pto.store_scalar(
                        expert_tokens_before_capacity_out_ptr,
                        const(expert),
                        c0_i32,
                    )
                for row_idx in range(c0, cTokens, c1):
                    expert_val = pto.index_cast(pto.load_scalar(i32, expert_idx_ptr, row_idx))
                    for expert in range(config.experts):
                        count = pto.index_cast(
                            pto.load_scalar(i32, expert_tokens_before_capacity_out_ptr, const(expert))
                        )
                        updated_count = pto.select(expert_val == const(expert), count + c1, count)
                        pto.store_scalar(
                            expert_tokens_before_capacity_out_ptr,
                            const(expert),
                            pto.index_cast(updated_count, i32),
                        )

                running = c0
                for expert in range(config.experts):
                    count = pto.index_cast(
                        pto.load_scalar(i32, expert_tokens_before_capacity_out_ptr, const(expert))
                    )
                    running = running + count
                    pto.store_scalar(
                        expert_tokens_count_or_cumsum_out_ptr,
                        const(expert),
                        pto.index_cast(running, i32),
                    )

    return moe_init_routing_v2_seed


class MoeInitRoutingV2Wrapper:
    def __init__(self, *, output_dir):
        self._output_dir = Path(output_dir)
        self._config = _config()
        self._kernel = _build_kernel(config=self._config, output_dir=self._output_dir)

    def _build(self):
        self._kernel._build()

    def _artifact_paths(self):
        return self._kernel._artifact_paths()

    @property
    def library_path(self):
        return getattr(self._kernel, "library_path", None)

    @property
    def output_dir(self):
        return str(self._output_dir)

    def __call__(
        self,
        expanded_x_out_ptr,
        expanded_row_idx_out_ptr,
        expert_tokens_count_or_cumsum_out_ptr,
        expert_tokens_before_capacity_out_ptr,
        x_ptr,
        expert_idx_ptr,
        stream_ptr=None,
    ):
        self._kernel(
            expanded_x_out_ptr,
            expanded_row_idx_out_ptr,
            expert_tokens_count_or_cumsum_out_ptr,
            expert_tokens_before_capacity_out_ptr,
            x_ptr,
            expert_idx_ptr,
            stream_ptr=stream_ptr,
        )


def build_jit_wrapper(*, output_dir):
    return MoeInitRoutingV2Wrapper(output_dir=output_dir)
