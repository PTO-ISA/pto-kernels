"""Constrained PTO-DSL seed for moe_finalize_routing_v2_grad on 910B."""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class MoeFinalizeRoutingV2GradConfig:
    tokens: int
    hidden: int
    experts: int
    block_dim: int


def _config() -> MoeFinalizeRoutingV2GradConfig:
    return MoeFinalizeRoutingV2GradConfig(
        tokens=tuned_int("PTO_MOE_FINALIZE_V2_GRAD_TOKENS", 16, valid_values=(16, 128, 256)),
        hidden=tuned_int("PTO_MOE_FINALIZE_V2_GRAD_HIDDEN", 16, valid_values=(16, 64, 128)),
        experts=tuned_int("PTO_MOE_FINALIZE_V2_GRAD_EXPERTS", 4, valid_values=(4,)),
        block_dim=tuned_int("PTO_MOE_FINALIZE_V2_GRAD_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)),
    )


def _meta_data(config: MoeFinalizeRoutingV2GradConfig):
    fp16 = pto.float16
    i32 = pto.int32

    ptr = pto.PtrType(fp16)
    ptr_i32 = pto.PtrType(i32)
    tensor = pto.TensorType(rank=2, dtype=fp16)
    sub_row = pto.SubTensorType(shape=[1, config.hidden], dtype=fp16)

    cfg = pto.TileConfig()
    row_tile = pto.TileType(shape=[1, config.hidden], dtype=fp16, memory_space="VEC", config=cfg)

    return {
        "ptr": ptr,
        "ptr_i32": ptr_i32,
        "tensor": tensor,
        "sub_row": sub_row,
        "row_tile": row_tile,
    }


def _build_finalize_v2_grad_kernel(*, config: MoeFinalizeRoutingV2GradConfig, output_dir):
    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.tokens, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_finalize_routing_v2_grad_seed(
        grad_expanded_x_out_ptr: "ptr",
        grad_scales_out_ptr: "ptr",
        grad_y_ptr: "ptr",
        expanded_row_idx_ptr: "ptr_i32",
        expanded_x_ptr: "ptr",
        scales_ptr: "ptr",
        expert_idx_ptr: "ptr_i32",
        bias_ptr: "ptr",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cTokens = const(config.tokens)
        cHidden = const(config.hidden)
        cExperts = const(config.experts)
        i32 = pto.int32
        fp16 = pto.float16

        tv_grad_y = pto.as_tensor(tensor, ptr=grad_y_ptr, shape=[cTokens, cHidden], strides=[cHidden, c1])
        tv_expanded_x = pto.as_tensor(tensor, ptr=expanded_x_ptr, shape=[cTokens, cHidden], strides=[cHidden, c1])
        tv_bias = pto.as_tensor(tensor, ptr=bias_ptr, shape=[cExperts, cHidden], strides=[cHidden, c1])
        tv_grad_expanded_x_out = pto.as_tensor(
            tensor, ptr=grad_expanded_x_out_ptr, shape=[cTokens, cHidden], strides=[cHidden, c1]
        )
        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cTokens, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cTokens)

            grad_y_row = pto.alloc_tile(row_tile)
            expanded_x_row = pto.alloc_tile(row_tile)
            bias_row = pto.alloc_tile(row_tile)
            scale_row = pto.alloc_tile(row_tile)
            grad_expanded_x_row = pto.alloc_tile(row_tile)

            for row_idx in range(row_start, row_end, c1):
                expanded_row_idx = pto.index_cast(pto.load_scalar(i32, expanded_row_idx_ptr, row_idx))
                expert_idx = pto.index_cast(pto.load_scalar(i32, expert_idx_ptr, row_idx))
                scale = pto.load_scalar(fp16, scales_ptr, row_idx)

                grad_y_view = pto.slice_view(sub_row, source=tv_grad_y, offsets=[row_idx, c0], sizes=[c1, cHidden])
                expanded_x_view = pto.slice_view(
                    sub_row, source=tv_expanded_x, offsets=[expanded_row_idx, c0], sizes=[c1, cHidden]
                )
                bias_view = pto.slice_view(sub_row, source=tv_bias, offsets=[expert_idx, c0], sizes=[c1, cHidden])
                grad_expanded_x_out_view = pto.slice_view(
                    sub_row, source=tv_grad_expanded_x_out, offsets=[expanded_row_idx, c0], sizes=[c1, cHidden]
                )

                pto.load(grad_y_view, grad_y_row)
                pto.load(expanded_x_view, expanded_x_row)
                pto.load(bias_view, bias_row)

                pto.expands(scale, scale_row)
                pto.mul(grad_y_row, scale_row, grad_expanded_x_row)
                pto.store(grad_expanded_x_row, grad_expanded_x_out_view)

                grad_scale = const(0.0, dtype=fp16)
                grad_y_base = row_idx * cHidden
                expanded_x_base = expanded_row_idx * cHidden
                bias_base = expert_idx * cHidden
                for col_idx_int in range(config.hidden):
                    col_idx = col_idx_int
                    grad_y_scalar = pto.wrap_value(pto.load_scalar(fp16, grad_y_ptr, grad_y_base + col_idx))
                    expanded_x_scalar = pto.wrap_value(pto.load_scalar(fp16, expanded_x_ptr, expanded_x_base + col_idx))
                    bias_scalar = pto.wrap_value(pto.load_scalar(fp16, bias_ptr, bias_base + col_idx))
                    grad_scale = grad_scale + (expanded_x_scalar + bias_scalar) * grad_y_scalar
                pto.store_scalar(grad_scales_out_ptr, row_idx, grad_scale)

    return moe_finalize_routing_v2_grad_seed


class MoeFinalizeRoutingV2GradWrapper:
    def __init__(self, *, output_dir):
        self._output_dir = Path(output_dir)
        self._config = _config()
        self._kernel = _build_finalize_v2_grad_kernel(config=self._config, output_dir=self._output_dir)

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
        grad_expanded_x_out_ptr,
        grad_scales_out_ptr,
        grad_y_ptr,
        expanded_row_idx_ptr,
        expanded_x_ptr,
        scales_ptr,
        expert_idx_ptr,
        bias_ptr,
        stream_ptr=None,
    ):
        self._kernel(
            grad_expanded_x_out_ptr,
            grad_scales_out_ptr,
            grad_y_ptr,
            expanded_row_idx_ptr,
            expanded_x_ptr,
            scales_ptr,
            expert_idx_ptr,
            bias_ptr,
            stream_ptr=stream_ptr,
        )


def build_jit_wrapper(*, output_dir):
    return MoeFinalizeRoutingV2GradWrapper(output_dir=output_dir)
