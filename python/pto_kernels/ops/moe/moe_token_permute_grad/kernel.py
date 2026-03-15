"""Constrained PTO-DSL seed for moe_token_permute_grad on 910B."""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class MoeTokenPermuteGradConfig:
    tokens: int
    hidden: int
    topk: int
    block_dim: int

    @property
    def total_rows(self) -> int:
        return self.tokens * self.topk


def _config() -> MoeTokenPermuteGradConfig:
    return MoeTokenPermuteGradConfig(
        tokens=tuned_int("PTO_MOE_PERMUTE_GRAD_TOKENS", 8, valid_values=(8, 128, 256)),
        hidden=tuned_int("PTO_MOE_PERMUTE_GRAD_HIDDEN", 16, valid_values=(16, 64, 128)),
        topk=tuned_int("PTO_MOE_PERMUTE_GRAD_TOPK", 1, valid_values=(1,)),
        block_dim=tuned_int("PTO_MOE_PERMUTE_GRAD_BLOCK_DIM", 20, valid_values=(1, 2, 4, 8, 16, 20)),
    )


def _meta_data(config: MoeTokenPermuteGradConfig):
    dtype = pto.float16
    i32 = pto.int32

    ptr = pto.PtrType(dtype)
    ptr_i32 = pto.PtrType(i32)
    tensor = pto.TensorType(rank=2, dtype=dtype)
    tensor_i32 = pto.TensorType(rank=1, dtype=i32)
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
        "tensor_i32": tensor_i32,
        "sub_row": sub_row,
        "tile_row": tile_row,
    }


def _build_kernel(*, config: MoeTokenPermuteGradConfig, output_dir):
    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.tokens, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_token_permute_grad_seed(out_ptr: "ptr", grad_perm_ptr: "ptr", sorted_indices_ptr: "ptr_i32") -> None:
        c0 = const(0)
        c1 = const(1)
        cTokens = const(config.tokens)
        cHidden = const(config.hidden)
        i32 = pto.int32

        tv_grad = pto.as_tensor(
            tensor,
            ptr=grad_perm_ptr,
            shape=[const(config.total_rows), cHidden],
            strides=[cHidden, c1],
        )
        tv_out = pto.as_tensor(
            tensor,
            ptr=out_ptr,
            shape=[cTokens, cHidden],
            strides=[cHidden, c1],
        )
        _ = pto.as_tensor(
            tensor_i32,
            ptr=sorted_indices_ptr,
            shape=[const(config.total_rows)],
            strides=[c1],
        )

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cTokens, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cTokens)

            acc_row = pto.alloc_tile(tile_row)
            for row_idx in range(row_start, row_end, c1):
                first_src_idx = pto.index_cast(pto.load_scalar(i32, sorted_indices_ptr, row_idx))
                first_view = pto.slice_view(
                    sub_row,
                    source=tv_grad,
                    offsets=[first_src_idx, c0],
                    sizes=[c1, cHidden],
                )
                out_view = pto.slice_view(
                    sub_row,
                    source=tv_out,
                    offsets=[row_idx, c0],
                    sizes=[c1, cHidden],
                )
                pto.load(first_view, acc_row)
                pto.store(acc_row, out_view)

    return moe_token_permute_grad_seed


class MoeTokenPermuteGradWrapper:
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

    def __call__(self, out_ptr, grad_perm_ptr, sorted_indices_ptr, stream_ptr=None):
        self._kernel(out_ptr, grad_perm_ptr, sorted_indices_ptr, stream_ptr=stream_ptr)


def build_jit_wrapper(*, output_dir):
    return MoeTokenPermuteGradWrapper(output_dir=output_dir)
