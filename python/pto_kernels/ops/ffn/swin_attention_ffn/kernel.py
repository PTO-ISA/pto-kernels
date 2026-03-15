"""Constrained PTO rewrite of SwinAttentionFFN for the zero-shift 910B slice."""

import os
from pathlib import Path

from ptodsl import jit, pto

from pto_kernels.ops.ffn.common import build_matmul_stage


const = pto.const


class SwinAttentionFfnConfig:
    def __init__(self, *, tokens: int, hidden: int, base_m: int, base_n: int, base_k: int, block_dim: int, add_block_dim: int):
        self.tokens = tokens
        self.hidden = hidden
        self.base_m = base_m
        self.base_n = base_n
        self.base_k = base_k
        self.block_dim = block_dim
        self.add_block_dim = add_block_dim

    def validate(self) -> None:
        for axis_name, axis, base in (
            ("tokens", self.tokens, self.base_m),
            ("hidden", self.hidden, self.base_n),
            ("hidden", self.hidden, self.base_k),
        ):
            if axis % base != 0:
                raise ValueError(f"swin_attention_ffn zero-shift slice requires {axis_name}={axis} divisible by {base}")


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def make_config() -> SwinAttentionFfnConfig:
    return SwinAttentionFfnConfig(
        tokens=_int_env("PTO_SWIN_ATTENTION_FFN_TOKENS", 3072),
        hidden=_int_env("PTO_SWIN_ATTENTION_FFN_HIDDEN", 128),
        base_m=_int_env("PTO_SWIN_ATTENTION_FFN_BASE_M", 128),
        base_n=_int_env("PTO_SWIN_ATTENTION_FFN_BASE_N", 128),
        base_k=_int_env("PTO_SWIN_ATTENTION_FFN_BASE_K", 128),
        block_dim=_int_env("PTO_SWIN_ATTENTION_FFN_BLOCK_DIM", 24),
        add_block_dim=_int_env("PTO_SWIN_ATTENTION_FFN_ADD_BLOCK_DIM", 24),
    )


def _launch_block_dim(total_rows: int, base_m: int, requested: int) -> int:
    total_tiles = total_rows // base_m
    return max(1, min(total_tiles, requested))


def _add_meta_data(hidden: int):
    dtype = pto.float16
    ptr = pto.PtrType(dtype)
    tensor2 = pto.TensorType(rank=2, dtype=dtype)
    row_view = pto.SubTensorType(shape=[1, hidden], dtype=dtype)
    row_tile = pto.TileType(shape=[1, hidden], dtype=dtype, memory_space="VEC")
    return {
        "ptr": ptr,
        "tensor2": tensor2,
        "row_view": row_view,
        "row_tile": row_tile,
    }


def build_add_stage(*, config: SwinAttentionFfnConfig, output_dir: Path):
    hidden = config.hidden

    @jit(
        meta_data=lambda: _add_meta_data(hidden),
        output_dir=output_dir,
        block_dim=_launch_block_dim(config.tokens, config.base_m, config.add_block_dim),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def _stage(out_ptr: "ptr", tmp_ptr: "ptr", bias_ptr: "ptr", x3_ptr: "ptr") -> None:
        c0 = const(0)
        c1 = const(1)
        cHidden = const(config.hidden)
        cTokens = const(config.tokens)

        tv_tmp = pto.as_tensor(tensor2, ptr=tmp_ptr, shape=[cTokens, cHidden], strides=[cHidden, c1])
        tv_x3 = pto.as_tensor(tensor2, ptr=x3_ptr, shape=[cTokens, cHidden], strides=[cHidden, c1])
        tv_out = pto.as_tensor(tensor2, ptr=out_ptr, shape=[cTokens, cHidden], strides=[cHidden, c1])
        tv_bias = pto.as_tensor(tensor2, ptr=bias_ptr, shape=[c1, cHidden], strides=[cHidden, c1])

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cTokens, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cTokens)

            tmp_tile = pto.alloc_tile(row_tile)
            x3_tile = pto.alloc_tile(row_tile)
            bias_tile = pto.alloc_tile(row_tile)
            sum_tile = pto.alloc_tile(row_tile)
            out_tile = pto.alloc_tile(row_tile)

            bias_view_ref = pto.slice_view(row_view, source=tv_bias, offsets=[c0, c0], sizes=[c1, cHidden])
            pto.load(bias_view_ref, bias_tile)

            for row_idx in range(row_start, row_end, c1):
                tmp_view_ref = pto.slice_view(row_view, source=tv_tmp, offsets=[row_idx, c0], sizes=[c1, cHidden])
                x3_view_ref = pto.slice_view(row_view, source=tv_x3, offsets=[row_idx, c0], sizes=[c1, cHidden])
                out_view_ref = pto.slice_view(row_view, source=tv_out, offsets=[row_idx, c0], sizes=[c1, cHidden])

                pto.load(tmp_view_ref, tmp_tile)
                pto.load(x3_view_ref, x3_tile)
                pto.add(tmp_tile, x3_tile, sum_tile)
                pto.add(sum_tile, bias_tile, out_tile)
                pto.store(out_tile, out_view_ref)

    return _stage


class SwinAttentionFfnWrapper:
    def __init__(self, *, config: SwinAttentionFfnConfig, output_dir: Path):
        config.validate()
        self._config = config
        self._output_dir = Path(output_dir)
        self._stage1 = build_matmul_stage(
            config=config,
            output_dir=self._output_dir / "stage1",
            stage_name="swin_attention_ffn_stage1",
            input_m=config.tokens,
            input_k=config.hidden,
            input_n=config.hidden,
            base_m=config.base_m,
            base_n=config.base_n,
            base_k=config.base_k,
            stage_block_dim=config.block_dim,
        )
        self._stage2 = build_add_stage(config=config, output_dir=self._output_dir / "stage2_add")

    def _build(self):
        self._stage1._build()
        self._stage2._build()
        return self

    def __call__(self, out_ptr, tmp_ptr, x1_ptr, x2_ptr, bias_ptr, x3_ptr):
        self._stage1(tmp_ptr, x1_ptr, x2_ptr)
        self._stage2(out_ptr, tmp_ptr, bias_ptr, x3_ptr)

    def _artifact_paths(self):
        paths = []
        for stage in (self._stage1, self._stage2):
            paths.extend(stage._artifact_paths())
        return tuple(paths)

    @property
    def library_path(self):
        return str(self._output_dir / "stage2_add" / "kernel.so")


def build_jit_wrapper(output_dir) -> SwinAttentionFfnWrapper:
    return SwinAttentionFfnWrapper(config=make_config(), output_dir=Path(output_dir))
