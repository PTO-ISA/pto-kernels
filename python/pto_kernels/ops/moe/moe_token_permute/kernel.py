"""Constrained PTO-DSL seed for moe_token_permute on 910B.

The phase-1 seed focuses on the data-movement part of top-1 token permutation.
It does not sort indices on-device yet. Instead, the benchmark harness provides
the flattened gather map for the current input permutation, and the PTO seed
uses that map to reorder tokens plus copy the permutation metadata.
"""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class MoeTokenPermuteConfig:
    tokens: int
    hidden: int
    block_dim: int

    @property
    def total(self) -> int:
        return self.tokens * self.hidden


def _config() -> MoeTokenPermuteConfig:
    return MoeTokenPermuteConfig(
        tokens=tuned_int("PTO_MOE_TOKENS", 8, valid_values=(8, 16)),
        hidden=tuned_int("PTO_MOE_HIDDEN", 16, valid_values=(16, 32)),
        block_dim=tuned_int("PTO_MOE_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)),
    )


def _permute_meta_data(config: MoeTokenPermuteConfig):
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


def _copy_indices_meta_data(config: MoeTokenPermuteConfig):
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


def _build_permute_kernel(*, config: MoeTokenPermuteConfig, output_dir):
    @jit(
        meta_data=lambda: _permute_meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.tokens, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_token_permute_seed(out_ptr: "ptr", tokens_ptr: "ptr", gather_ptr: "ptr_i32") -> None:
        c0 = const(0)
        c1 = const(1)
        cTokens = const(config.tokens)
        cHidden = const(config.hidden)
        cTotal = const(config.total)

        tv_tokens = pto.as_tensor(
            tensor,
            ptr=tokens_ptr,
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

    return moe_token_permute_seed


def _build_copy_indices_kernel(*, config: MoeTokenPermuteConfig, output_dir):
    @jit(
        meta_data=lambda: _copy_indices_meta_data(config),
        output_dir=output_dir,
        block_dim=1,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_token_permute_copy_indices(out_ptr: "ptr_i32", indices_ptr: "ptr_i32") -> None:
        c0 = const(0)
        c1 = const(1)
        cTokens = const(config.tokens)

        tv_in = pto.as_tensor(
            tensor_i32,
            ptr=indices_ptr,
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
            tb_in = pto.alloc_tile(tile_i32)
            sv_in = pto.slice_view(sub_i32, source=tv_in, offsets=[c0], sizes=[cTokens])
            sv_out = pto.slice_view(sub_i32, source=tv_out, offsets=[c0], sizes=[cTokens])
            pto.load(sv_in, tb_in)
            pto.store(tb_in, sv_out)

    return moe_token_permute_copy_indices


class MoeTokenPermuteWrapper:
    def __init__(self, *, output_dir):
        self._output_dir = Path(output_dir)
        self._config = _config()
        self._permute = _build_permute_kernel(
            config=self._config,
            output_dir=self._output_dir / "stage1_permute",
        )
        self._copy = _build_copy_indices_kernel(
            config=self._config,
            output_dir=self._output_dir / "stage2_copy_indices",
        )

    def _build(self):
        self._permute._build()
        self._copy._build()

    def _artifact_paths(self):
        return (
            *self._permute._artifact_paths(),
            *self._copy._artifact_paths(),
        )

    @property
    def library_path(self):
        return None

    @property
    def output_dir(self):
        return str(self._output_dir)

    def __call__(self, out_ptr, sorted_indices_out_ptr, tokens_ptr, indices_ptr, gather_ptr, stream_ptr=None):
        self._permute(out_ptr, tokens_ptr, gather_ptr, stream_ptr=stream_ptr)
        self._copy(sorted_indices_out_ptr, indices_ptr, stream_ptr=stream_ptr)


def build_jit_wrapper(*, output_dir):
    return MoeTokenPermuteWrapper(output_dir=output_dir)
