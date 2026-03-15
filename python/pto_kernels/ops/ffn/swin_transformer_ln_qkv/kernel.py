"""Constrained PTO rewrite of SwinTransformerLnQKV for the 910B fp16 slice."""

import os
from pathlib import Path

from ptodsl import jit, pto

from pto_kernels.ops.ffn.common import build_matmul_stage


const = pto.const
EPS = 1e-5


class SwinTransformerLnQkvConfig:
    def __init__(
        self,
        *,
        tokens: int,
        hidden: int,
        heads: int,
        head_dim: int,
        base_m: int,
        base_n: int,
        base_k: int,
        layernorm_block_dim: int,
        matmul_block_dim: int,
        split_block_dim: int,
    ):
        self.tokens = tokens
        self.hidden = hidden
        self.heads = heads
        self.head_dim = head_dim
        self.base_m = base_m
        self.base_n = base_n
        self.base_k = base_k
        self.layernorm_block_dim = layernorm_block_dim
        self.matmul_block_dim = matmul_block_dim
        self.split_block_dim = split_block_dim

    @property
    def qkv_hidden(self) -> int:
        return self.hidden * 3

    def validate(self) -> None:
        if self.hidden != self.heads * self.head_dim:
            raise ValueError("swin_transformer_ln_qkv requires hidden == heads * head_dim")
        for axis_name, axis, base in (
            ("tokens", self.tokens, self.base_m),
            ("hidden", self.hidden, self.base_k),
            ("qkv_hidden", self.qkv_hidden, self.base_n),
        ):
            if axis % base != 0:
                raise ValueError(
                    f"swin_transformer_ln_qkv slice requires {axis_name}={axis} divisible by {base}"
                )


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def make_config() -> SwinTransformerLnQkvConfig:
    return SwinTransformerLnQkvConfig(
        tokens=_int_env("PTO_SWIN_LN_QKV_TOKENS", 2048),
        hidden=_int_env("PTO_SWIN_LN_QKV_HIDDEN", 128),
        heads=_int_env("PTO_SWIN_LN_QKV_HEADS", 4),
        head_dim=_int_env("PTO_SWIN_LN_QKV_HEAD_DIM", 32),
        base_m=_int_env("PTO_SWIN_LN_QKV_BASE_M", 128),
        base_n=_int_env("PTO_SWIN_LN_QKV_BASE_N", 128),
        base_k=_int_env("PTO_SWIN_LN_QKV_BASE_K", 64),
        layernorm_block_dim=_int_env("PTO_SWIN_LN_QKV_LN_BLOCK_DIM", 24),
        matmul_block_dim=_int_env("PTO_SWIN_LN_QKV_MATMUL_BLOCK_DIM", 24),
        split_block_dim=_int_env("PTO_SWIN_LN_QKV_SPLIT_BLOCK_DIM", 24),
    )


def _launch_block_dim(total_rows: int, requested: int) -> int:
    return max(1, min(total_rows, requested))


def _layernorm_meta_data(hidden: int):
    fp16 = pto.float16
    fp32 = pto.float32
    ptr = pto.PtrType(fp16)
    tensor = pto.TensorType(rank=2, dtype=fp16)
    row_view = pto.SubTensorType(shape=[1, hidden], dtype=fp16)
    row_tile = pto.TileType(shape=[1, hidden], dtype=fp16, memory_space="VEC")
    row_tile_acc = pto.TileType(shape=[1, hidden], dtype=fp32, memory_space="VEC")
    scalar_tile = pto.TileType(
        shape=[1, hidden],
        valid_shape=[1, 1],
        dtype=fp32,
        memory_space="VEC",
        config=pto.TileConfig(),
    )
    return {
        "ptr": ptr,
        "tensor": tensor,
        "row_view": row_view,
        "row_tile": row_tile,
        "row_tile_acc": row_tile_acc,
        "scalar_tile": scalar_tile,
    }


def _split_meta_data(hidden: int):
    fp16 = pto.float16
    ptr = pto.PtrType(fp16)
    tensor_in = pto.TensorType(rank=2, dtype=fp16)
    tensor_out = pto.TensorType(rank=2, dtype=fp16)
    row_view = pto.SubTensorType(shape=[1, hidden], dtype=fp16)
    row_tile = pto.TileType(shape=[1, hidden], dtype=fp16, memory_space="VEC")
    return {
        "ptr": ptr,
        "tensor_in": tensor_in,
        "tensor_out": tensor_out,
        "row_view": row_view,
        "row_tile": row_tile,
    }


def build_layernorm_stage(*, config: SwinTransformerLnQkvConfig, output_dir: Path):
    hidden = config.hidden
    inv_hidden = 1.0 / float(hidden)

    @jit(
        meta_data=lambda: _layernorm_meta_data(hidden),
        output_dir=output_dir,
        block_dim=_launch_block_dim(config.tokens, config.layernorm_block_dim),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def _stage(out_ptr: "ptr", x_ptr: "ptr", gamma_ptr: "ptr", beta_ptr: "ptr") -> None:
        c0 = const(0)
        c1 = const(1)
        cHidden = const(config.hidden)
        cTokens = const(config.tokens)

        tv_out = pto.as_tensor(tensor, ptr=out_ptr, shape=[cTokens, cHidden], strides=[cHidden, c1])
        tv_x = pto.as_tensor(tensor, ptr=x_ptr, shape=[cTokens, cHidden], strides=[cHidden, c1])
        tv_gamma = pto.as_tensor(tensor, ptr=gamma_ptr, shape=[c1, cHidden], strides=[cHidden, c1])
        tv_beta = pto.as_tensor(tensor, ptr=beta_ptr, shape=[c1, cHidden], strides=[cHidden, c1])

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cTokens, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cTokens)

            x_tile = pto.alloc_tile(row_tile)
            gamma_tile = pto.alloc_tile(row_tile)
            beta_tile = pto.alloc_tile(row_tile)
            x_acc = pto.alloc_tile(row_tile_acc)
            gamma_acc = pto.alloc_tile(row_tile_acc)
            beta_acc = pto.alloc_tile(row_tile_acc)
            mean_acc = pto.alloc_tile(row_tile_acc)
            centered = pto.alloc_tile(row_tile_acc)
            sq = pto.alloc_tile(row_tile_acc)
            var_acc = pto.alloc_tile(row_tile_acc)
            inv_std = pto.alloc_tile(row_tile_acc)
            tmp = pto.alloc_tile(row_tile_acc)
            refine = pto.alloc_tile(row_tile_acc)
            const_row = pto.alloc_tile(row_tile_acc)
            norm = pto.alloc_tile(row_tile_acc)
            out_acc = pto.alloc_tile(row_tile_acc)
            scalar = pto.alloc_tile(scalar_tile)

            gamma_view = pto.slice_view(row_view, source=tv_gamma, offsets=[c0, c0], sizes=[c1, cHidden])
            beta_view = pto.slice_view(row_view, source=tv_beta, offsets=[c0, c0], sizes=[c1, cHidden])
            pto.load(gamma_view, gamma_tile)
            pto.load(beta_view, beta_tile)
            pto.cvt(gamma_tile, gamma_acc)
            pto.cvt(beta_tile, beta_acc)

            for row_idx in range(row_start, row_end, c1):
                x_view = pto.slice_view(row_view, source=tv_x, offsets=[row_idx, c0], sizes=[c1, cHidden])
                out_view = pto.slice_view(row_view, source=tv_out, offsets=[row_idx, c0], sizes=[c1, cHidden])

                pto.load(x_view, x_tile)
                pto.cvt(x_tile, x_acc)

                pto.row_sum(x_acc, tmp, scalar)
                pto.row_expand(scalar, mean_acc)
                pto.muls(mean_acc, const(inv_hidden, dtype=pto.float32), mean_acc)
                pto.sub(x_acc, mean_acc, centered)

                pto.mul(centered, centered, sq)
                pto.row_sum(sq, tmp, scalar)
                pto.row_expand(scalar, var_acc)
                pto.muls(var_acc, const(inv_hidden, dtype=pto.float32), var_acc)
                pto.adds(var_acc, const(EPS, dtype=pto.float32), var_acc)

                pto.rsqrt(var_acc, inv_std)
                pto.mul(inv_std, inv_std, refine)
                pto.mul(var_acc, refine, refine)
                pto.muls(refine, const(0.5, dtype=pto.float32), refine)
                pto.expands(const(1.5, dtype=pto.float32), const_row)
                pto.sub(const_row, refine, refine)
                pto.mul(inv_std, refine, inv_std)

                pto.mul(centered, inv_std, norm)
                pto.mul(norm, gamma_acc, out_acc)
                pto.add(out_acc, beta_acc, out_acc)
                pto.cvt(out_acc, x_tile)
                pto.store(x_tile, out_view)

    return _stage


def build_split_stage(*, config: SwinTransformerLnQkvConfig, output_dir: Path):
    hidden = config.hidden

    @jit(
        meta_data=lambda: _split_meta_data(hidden),
        output_dir=output_dir,
        block_dim=_launch_block_dim(config.tokens, config.split_block_dim),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def _stage(q_ptr: "ptr", k_ptr: "ptr", v_ptr: "ptr", packed_ptr: "ptr", bias_ptr: "ptr") -> None:
        c0 = const(0)
        c1 = const(1)
        cHidden = const(config.hidden)
        cHidden2 = cHidden + cHidden
        cQkvHidden = const(config.qkv_hidden)
        cTokens = const(config.tokens)

        tv_q = pto.as_tensor(tensor_out, ptr=q_ptr, shape=[cTokens, cHidden], strides=[cHidden, c1])
        tv_k = pto.as_tensor(tensor_out, ptr=k_ptr, shape=[cTokens, cHidden], strides=[cHidden, c1])
        tv_v = pto.as_tensor(tensor_out, ptr=v_ptr, shape=[cTokens, cHidden], strides=[cHidden, c1])
        tv_packed = pto.as_tensor(tensor_in, ptr=packed_ptr, shape=[cTokens, cQkvHidden], strides=[cQkvHidden, c1])
        tv_bias = pto.as_tensor(tensor_in, ptr=bias_ptr, shape=[c1, cQkvHidden], strides=[cQkvHidden, c1])

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cTokens, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cTokens)

            q_row_tile = pto.alloc_tile(row_tile)
            k_row_tile = pto.alloc_tile(row_tile)
            v_row_tile = pto.alloc_tile(row_tile)
            q_bias_tile = pto.alloc_tile(row_tile)
            k_bias_tile = pto.alloc_tile(row_tile)
            v_bias_tile = pto.alloc_tile(row_tile)

            q_bias_view = pto.slice_view(row_view, source=tv_bias, offsets=[c0, c0], sizes=[c1, cHidden])
            k_bias_view = pto.slice_view(row_view, source=tv_bias, offsets=[c0, cHidden], sizes=[c1, cHidden])
            v_bias_view = pto.slice_view(row_view, source=tv_bias, offsets=[c0, cHidden2], sizes=[c1, cHidden])
            pto.load(q_bias_view, q_bias_tile)
            pto.load(k_bias_view, k_bias_tile)
            pto.load(v_bias_view, v_bias_tile)

            for row_idx in range(row_start, row_end, c1):
                q_packed_view = pto.slice_view(row_view, source=tv_packed, offsets=[row_idx, c0], sizes=[c1, cHidden])
                k_packed_view = pto.slice_view(
                    row_view, source=tv_packed, offsets=[row_idx, cHidden], sizes=[c1, cHidden]
                )
                v_packed_view = pto.slice_view(
                    row_view, source=tv_packed, offsets=[row_idx, cHidden2], sizes=[c1, cHidden]
                )
                q_view_ref = pto.slice_view(row_view, source=tv_q, offsets=[row_idx, c0], sizes=[c1, cHidden])
                k_view_ref = pto.slice_view(row_view, source=tv_k, offsets=[row_idx, c0], sizes=[c1, cHidden])
                v_view_ref = pto.slice_view(row_view, source=tv_v, offsets=[row_idx, c0], sizes=[c1, cHidden])

                pto.load(q_packed_view, q_row_tile)
                pto.load(k_packed_view, k_row_tile)
                pto.load(v_packed_view, v_row_tile)
                pto.add(q_row_tile, q_bias_tile, q_row_tile)
                pto.add(k_row_tile, k_bias_tile, k_row_tile)
                pto.add(v_row_tile, v_bias_tile, v_row_tile)
                pto.store(q_row_tile, q_view_ref)
                pto.store(k_row_tile, k_view_ref)
                pto.store(v_row_tile, v_view_ref)

    return _stage


class SwinTransformerLnQkvWrapper:
    def __init__(self, *, config: SwinTransformerLnQkvConfig, output_dir: Path):
        config.validate()
        self._config = config
        self._output_dir = Path(output_dir)
        self._layernorm = build_layernorm_stage(config=config, output_dir=self._output_dir / "stage1_layernorm")
        self._matmul = build_matmul_stage(
            config=config,
            output_dir=self._output_dir / "stage2_matmul",
            stage_name="swin_transformer_ln_qkv_stage2_matmul",
            input_m=config.tokens,
            input_k=config.hidden,
            input_n=config.qkv_hidden,
            base_m=config.base_m,
            base_n=config.base_n,
            base_k=config.base_k,
            stage_block_dim=config.matmul_block_dim,
        )
        self._split = build_split_stage(config=config, output_dir=self._output_dir / "stage3_split")

    def _build(self):
        self._layernorm._build()
        self._matmul._build()
        self._split._build()
        return self

    def __call__(
        self,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        packed_tmp_ptr,
        ln_tmp_ptr,
        x_ptr,
        gamma_ptr,
        beta_ptr,
        weight_ptr,
        bias_ptr,
    ):
        self._layernorm(ln_tmp_ptr, x_ptr, gamma_ptr, beta_ptr)
        self._matmul(packed_tmp_ptr, ln_tmp_ptr, weight_ptr)
        self._split(q_out_ptr, k_out_ptr, v_out_ptr, packed_tmp_ptr, bias_ptr)
        return q_out_ptr, k_out_ptr, v_out_ptr

    def _artifact_paths(self):
        paths = []
        for stage in (self._layernorm, self._matmul, self._split):
            paths.extend(stage._artifact_paths())
        return tuple(paths)

    @property
    def library_path(self):
        return str(self._output_dir / "stage3_split" / "kernel.so")


def build_jit_wrapper(*, output_dir):
    return SwinTransformerLnQkvWrapper(config=make_config(), output_dir=Path(output_dir))
