"""Constrained PTO-DSL seed for moe_re_routing on 910B."""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class MoeReRoutingConfig:
    tokens: int
    hidden: int
    ranks: int
    experts: int
    block_dim: int


def _config() -> MoeReRoutingConfig:
    return MoeReRoutingConfig(
        tokens=tuned_int("PTO_MOE_RE_ROUTING_TOKENS", 8, valid_values=(8, 128, 256)),
        hidden=tuned_int("PTO_MOE_RE_ROUTING_HIDDEN", 16, valid_values=(16, 64, 128)),
        ranks=tuned_int("PTO_MOE_RE_ROUTING_RANKS", 2, valid_values=(2, 4)),
        experts=tuned_int("PTO_MOE_RE_ROUTING_EXPERTS", 2, valid_values=(2, 4)),
        block_dim=tuned_int("PTO_MOE_RE_ROUTING_BLOCK_DIM", 20, valid_values=(1, 2, 4, 8, 16, 20)),
    )


def _meta_data(config: MoeReRoutingConfig):
    dtype = pto.float16
    f32 = pto.float32
    i32 = pto.int32

    ptr = pto.PtrType(dtype)
    ptr_f32 = pto.PtrType(f32)
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
        "ptr_f32": ptr_f32,
        "ptr_i32": ptr_i32,
        "tensor": tensor,
        "sub_row": sub_row,
        "tile_row": tile_row,
    }


def _build_kernel(*, config: MoeReRoutingConfig, output_dir):
    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.tokens, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_re_routing_seed(
        out_tokens_ptr: "ptr",
        out_scales_ptr: "ptr_f32",
        out_idx_ptr: "ptr_i32",
        out_expert_token_num_ptr: "ptr_i32",
        tokens_ptr: "ptr",
        counts_ptr: "ptr_i32",
        scales_ptr: "ptr_f32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cTokens = const(config.tokens)
        cHidden = const(config.hidden)
        cRanks = const(config.ranks)
        cExperts = const(config.experts)
        i32 = pto.int32
        f32 = pto.float32

        tv_tokens = pto.as_tensor(
            tensor,
            ptr=tokens_ptr,
            shape=[cTokens, cHidden],
            strides=[cHidden, c1],
        )
        tv_out_tokens = pto.as_tensor(
            tensor,
            ptr=out_tokens_ptr,
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

            if bid == c0:
                for expert in range(c0, cExperts, c1):
                    expert_total = c0
                    for rank in range(c0, cRanks, c1):
                        count_off = rank * cExperts + expert
                        count = pto.index_cast(pto.load_scalar(i32, counts_ptr, count_off))
                        expert_total = expert_total + count
                    pto.store_scalar(out_expert_token_num_ptr, expert, pto.index_cast(expert_total, pto.int32))

            for dst_row in range(row_start, row_end, c1):
                expert_idx = c0
                local_in_expert = c0
                expert_prefix = c0

                for expert in range(c0, cExperts, c1):
                    expert_total = c0
                    for rank in range(c0, cRanks, c1):
                        count_off = rank * cExperts + expert
                        count = pto.index_cast(pto.load_scalar(i32, counts_ptr, count_off))
                        expert_total = expert_total + count
                    expert_take = (dst_row >= expert_prefix) & (dst_row < expert_prefix + expert_total)
                    expert_idx = pto.select(expert_take, expert, expert_idx)
                    local_in_expert = pto.select(expert_take, dst_row - expert_prefix, local_in_expert)
                    expert_prefix = expert_prefix + expert_total

                rank_idx = c0
                local_offset = c0
                rank_prefix = c0
                for rank in range(c0, cRanks, c1):
                    count_off = rank * cExperts + expert_idx
                    count = pto.index_cast(pto.load_scalar(i32, counts_ptr, count_off))
                    rank_take = (local_in_expert >= rank_prefix) & (local_in_expert < rank_prefix + count)
                    rank_idx = pto.select(rank_take, rank, rank_idx)
                    local_offset = pto.select(rank_take, local_in_expert - rank_prefix, local_offset)
                    rank_prefix = rank_prefix + count

                src_prefix = c0
                for rank in range(c0, cRanks, c1):
                    for expert in range(c0, cExperts, c1):
                        count_off = rank * cExperts + expert
                        count = pto.index_cast(pto.load_scalar(i32, counts_ptr, count_off))
                        add_count = (rank < rank_idx) | ((rank == rank_idx) & (expert < expert_idx))
                        src_prefix = src_prefix + pto.select(add_count, count, c0)

                src_idx = src_prefix + local_offset
                src_view = pto.slice_view(sub_row, source=tv_tokens, offsets=[src_idx, c0], sizes=[c1, cHidden])
                dst_view = pto.slice_view(sub_row, source=tv_out_tokens, offsets=[dst_row, c0], sizes=[c1, cHidden])
                pto.load(src_view, token_row)
                pto.store(token_row, dst_view)
                pto.store_scalar(out_scales_ptr, dst_row, pto.load_scalar(f32, scales_ptr, src_idx))
                pto.store_scalar(out_idx_ptr, dst_row, pto.index_cast(src_idx, pto.int32))

    return moe_re_routing_seed


class MoeReRoutingWrapper:
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
        out_tokens_ptr,
        out_scales_ptr,
        out_idx_ptr,
        out_expert_token_num_ptr,
        tokens_ptr,
        counts_ptr,
        scales_ptr,
        stream_ptr=None,
    ):
        self._kernel(
            out_tokens_ptr,
            out_scales_ptr,
            out_idx_ptr,
            out_expert_token_num_ptr,
            tokens_ptr,
            counts_ptr,
            scales_ptr,
            stream_ptr=stream_ptr,
        )


def build_jit_wrapper(*, output_dir):
    return MoeReRoutingWrapper(output_dir=output_dir)
