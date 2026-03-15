"""Constrained PTO-DSL seed for moe_token_permute_with_routing_map_grad on 910B."""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class MoeTokenPermuteWithRoutingMapGradConfig:
    tokens: int
    hidden: int
    block_dim: int


def _config() -> MoeTokenPermuteWithRoutingMapGradConfig:
    return MoeTokenPermuteWithRoutingMapGradConfig(
        tokens=tuned_int("PTO_MOE_PERMUTE_ROUTING_MAP_GRAD_TOKENS", 8, valid_values=(8, 128, 256)),
        hidden=tuned_int("PTO_MOE_PERMUTE_ROUTING_MAP_GRAD_HIDDEN", 16, valid_values=(16, 64, 128)),
        block_dim=tuned_int(
            "PTO_MOE_PERMUTE_ROUTING_MAP_GRAD_BLOCK_DIM",
            20,
            valid_values=(1, 2, 4, 8, 16, 20),
        ),
    )


def _meta_data(config: MoeTokenPermuteWithRoutingMapGradConfig):
    dtype = pto.float16
    i32 = pto.int32

    ptr = pto.PtrType(dtype)
    ptr_i32 = pto.PtrType(i32)
    tensor = pto.TensorType(rank=2, dtype=dtype)
    row_view = pto.SubTensorType(shape=[1, config.hidden], dtype=dtype)

    cfg = pto.TileConfig()
    row_tile = pto.TileType(
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
        "row_view": row_view,
        "row_tile": row_tile,
    }


def _build_kernel(*, config: MoeTokenPermuteWithRoutingMapGradConfig, output_dir):
    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.tokens, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_token_permute_with_routing_map_grad_seed(
        token_grad_ptr: "ptr",
        probs_grad_ptr: "ptr",
        permuted_grad_ptr: "ptr",
        sorted_indices_ptr: "ptr_i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cTokens = const(config.tokens)
        cHidden = const(config.hidden)
        tv_in = pto.as_tensor(tensor, ptr=permuted_grad_ptr, shape=[cTokens, cHidden], strides=[cHidden, c1])
        tv_out = pto.as_tensor(tensor, ptr=token_grad_ptr, shape=[cTokens, cHidden], strides=[cHidden, c1])

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cTokens, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cTokens)
            in_row = pto.alloc_tile(row_tile)

            if bid == c0:
                pto.store_scalar(probs_grad_ptr, c0, const(0.0, dtype=pto.float16))

            for row_idx in range(row_start, row_end, c1):
                dst_row = pto.index_cast(pto.load_scalar(pto.int32, sorted_indices_ptr, row_idx))
                in_view = pto.slice_view(row_view, source=tv_in, offsets=[row_idx, c0], sizes=[c1, cHidden])
                out_view = pto.slice_view(row_view, source=tv_out, offsets=[dst_row, c0], sizes=[c1, cHidden])
                pto.load(in_view, in_row)
                pto.store(in_row, out_view)

    return moe_token_permute_with_routing_map_grad_seed


class MoeTokenPermuteWithRoutingMapGradWrapper:
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

    def __call__(self, token_grad_ptr, probs_grad_ptr, permuted_grad_ptr, sorted_indices_ptr, stream_ptr=None):
        self._kernel(
            token_grad_ptr,
            probs_grad_ptr,
            permuted_grad_ptr,
            sorted_indices_ptr,
            stream_ptr=stream_ptr,
        )


def build_jit_wrapper(*, output_dir):
    return MoeTokenPermuteWithRoutingMapGradWrapper(output_dir=output_dir)
