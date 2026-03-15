"""Constrained PTO-DSL seed for moe_token_unpermute on 910B."""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class MoeTokenUnpermuteConfig:
    tokens: int
    hidden: int
    block_dim: int

    @property
    def total(self) -> int:
        return self.tokens * self.hidden


def _config() -> MoeTokenUnpermuteConfig:
    return MoeTokenUnpermuteConfig(
        tokens=tuned_int("PTO_MOE_UNPERMUTE_TOKENS", 8, valid_values=(8, 128, 256)),
        hidden=tuned_int("PTO_MOE_UNPERMUTE_HIDDEN", 16, valid_values=(16, 64, 128)),
        block_dim=tuned_int("PTO_MOE_UNPERMUTE_BLOCK_DIM", 20, valid_values=(1, 2, 4, 8, 16, 20)),
    )


def _meta_data(config: MoeTokenUnpermuteConfig):
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


def _build_unpermute_kernel(*, config: MoeTokenUnpermuteConfig, output_dir):
    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.tokens, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_token_unpermute_seed(out_ptr: "ptr", permuted_ptr: "ptr", gather_ptr: "ptr_i32") -> None:
        c0 = const(0)
        c1 = const(1)
        cTokens = const(config.tokens)
        cHidden = const(config.hidden)
        cTotal = const(config.total)

        tv_tokens = pto.as_tensor(
            tensor,
            ptr=permuted_ptr,
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
            tb_tokens = pto.alloc_tile(tile_tokens)
            tb_gather = pto.alloc_tile(tile_gather)
            tb_out = pto.alloc_tile(tile_out)

            sv_tokens = pto.slice_view(sub_tokens, source=tv_tokens, offsets=[c0], sizes=[cTotal])
            pto.load(sv_tokens, tb_tokens)

            for row_idx in range(row_start, row_end, c1):
                row_off = row_idx * cHidden
                sv_gather = pto.slice_view(sub_gather, source=tv_gather, offsets=[row_off], sizes=[cHidden])
                sv_out = pto.slice_view(sub_out, source=tv_out, offsets=[row_off], sizes=[cHidden])

                pto.load(sv_gather, tb_gather)
                pto.gather(tb_tokens, tb_out, tb_gather)
                pto.store(tb_out, sv_out)

    return moe_token_unpermute_seed


class MoeTokenUnpermuteWrapper:
    def __init__(self, *, output_dir):
        self._output_dir = Path(output_dir)
        self._config = _config()
        self._kernel = _build_unpermute_kernel(
            config=self._config,
            output_dir=self._output_dir,
        )

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

    def __call__(self, out_ptr, permuted_ptr, gather_ptr, stream_ptr=None):
        self._kernel(out_ptr, permuted_ptr, gather_ptr, stream_ptr=stream_ptr)


def build_jit_wrapper(*, output_dir):
    return MoeTokenUnpermuteWrapper(output_dir=output_dir)
