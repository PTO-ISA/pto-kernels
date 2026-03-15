"""Constrained PTO-DSL seed for moe_finalize_routing on 910B."""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class MoeFinalizeRoutingConfig:
    tokens: int
    hidden: int
    experts: int
    block_dim: int

    @property
    def total(self) -> int:
        return self.tokens * self.hidden

    @property
    def bias_total(self) -> int:
        return self.experts * self.hidden


def _config() -> MoeFinalizeRoutingConfig:
    return MoeFinalizeRoutingConfig(
        tokens=tuned_int("PTO_MOE_FINALIZE_TOKENS", 16, valid_values=(16, 128, 256)),
        hidden=tuned_int("PTO_MOE_FINALIZE_HIDDEN", 16, valid_values=(16, 64, 128)),
        experts=tuned_int("PTO_MOE_FINALIZE_EXPERTS", 4, valid_values=(4,)),
        block_dim=tuned_int("PTO_MOE_FINALIZE_BLOCK_DIM", 20, valid_values=(1, 2, 4, 8, 16, 20)),
    )


def _meta_data(config: MoeFinalizeRoutingConfig):
    dtype = pto.float16
    i32 = pto.int32

    ptr = pto.PtrType(dtype)
    ptr_i32 = pto.PtrType(i32)
    tensor = pto.TensorType(rank=2, dtype=dtype)
    sub_row = pto.SubTensorType(shape=[1, config.hidden], dtype=dtype)
    tensor_idx = pto.TensorType(rank=1, dtype=i32)

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
        "tensor_idx": tensor_idx,
        "sub_row": sub_row,
        "tile_row": tile_row,
    }


def _build_finalize_kernel(*, config: MoeFinalizeRoutingConfig, output_dir):
    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.tokens, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_finalize_routing_seed(
        out_ptr: "ptr",
        expanded_ptr: "ptr",
        x1_ptr: "ptr",
        bias_ptr: "ptr",
        scales_ptr: "ptr",
        expanded_row_idx_ptr: "ptr_i32",
        expanded_expert_idx_ptr: "ptr_i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cTokens = const(config.tokens)
        cHidden = const(config.hidden)
        i32 = pto.int32
        dtype = pto.float16

        tv_expanded = pto.as_tensor(tensor, ptr=expanded_ptr, shape=[cTokens, cHidden], strides=[cHidden, c1])
        tv_x1 = pto.as_tensor(tensor, ptr=x1_ptr, shape=[cTokens, cHidden], strides=[cHidden, c1])
        tv_bias = pto.as_tensor(
            tensor,
            ptr=bias_ptr,
            shape=[const(config.experts), cHidden],
            strides=[cHidden, c1],
        )
        tv_out = pto.as_tensor(tensor, ptr=out_ptr, shape=[cTokens, cHidden], strides=[cHidden, c1])

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cTokens, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cTokens)

            x1_row = pto.alloc_tile(tile_row)
            expanded_row = pto.alloc_tile(tile_row)
            bias_row = pto.alloc_tile(tile_row)
            scale_row = pto.alloc_tile(tile_row)
            tmp_row = pto.alloc_tile(tile_row)
            out_row = pto.alloc_tile(tile_row)

            for row_idx in range(row_start, row_end, c1):
                expanded_row_idx = pto.index_cast(pto.load_scalar(i32, expanded_row_idx_ptr, row_idx))
                expanded_expert_idx = pto.index_cast(pto.load_scalar(i32, expanded_expert_idx_ptr, row_idx))
                scale = pto.load_scalar(dtype, scales_ptr, row_idx)

                x1_view = pto.slice_view(sub_row, source=tv_x1, offsets=[row_idx, c0], sizes=[c1, cHidden])
                out_view = pto.slice_view(sub_row, source=tv_out, offsets=[row_idx, c0], sizes=[c1, cHidden])
                expanded_view = pto.slice_view(
                    sub_row,
                    source=tv_expanded,
                    offsets=[expanded_row_idx, c0],
                    sizes=[c1, cHidden],
                )
                bias_view = pto.slice_view(
                    sub_row,
                    source=tv_bias,
                    offsets=[expanded_expert_idx, c0],
                    sizes=[c1, cHidden],
                )
                pto.load(x1_view, x1_row)
                pto.load(expanded_view, expanded_row)
                pto.load(bias_view, bias_row)

                pto.add(expanded_row, bias_row, tmp_row)
                pto.expands(scale, scale_row)
                pto.mul(tmp_row, scale_row, tmp_row)
                pto.add(x1_row, tmp_row, out_row)
                pto.store(out_row, out_view)

    return moe_finalize_routing_seed


class MoeFinalizeRoutingWrapper:
    def __init__(self, *, output_dir):
        self._output_dir = Path(output_dir)
        self._config = _config()
        self._kernel = _build_finalize_kernel(config=self._config, output_dir=self._output_dir)

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
        out_ptr,
        expanded_ptr,
        x1_ptr,
        bias_ptr,
        scales_ptr,
        expanded_row_idx_ptr,
        expanded_expert_idx_ptr,
        stream_ptr=None,
    ):
        self._kernel(
            out_ptr,
            expanded_ptr,
            x1_ptr,
            bias_ptr,
            scales_ptr,
            expanded_row_idx_ptr,
            expanded_expert_idx_ptr,
            stream_ptr=stream_ptr,
        )


def build_jit_wrapper(*, output_dir):
    return MoeFinalizeRoutingWrapper(output_dir=output_dir)
