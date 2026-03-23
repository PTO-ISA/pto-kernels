"""Constrained PTO-DSL seed for recurrent_gated_delta_rule on 910B/A3.

This checked-in slice keeps the currently validated host contract:
- batch=1
- nv=nk=16
- gk=None
- actual_seq_lengths fixed to full sequence length
- ssm_state_indices fixed to arange(T)
- num_accepted_tokens fixed to ones

The hot path stays tile-first:
- cube `gemv` for `state @ key` and `state_new @ query`
- vector row tiles for the recurrent state update
- no scalar pointer loops in the compute path

The broader family is still not complete. General ragged token-to-state mapping
and grouped/reused state-slot routing remain follow-on work.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class RecurrentGatedDeltaRuleConfig:
    rows: int
    dim: int
    block_dim: int


def _config() -> RecurrentGatedDeltaRuleConfig:
    return RecurrentGatedDeltaRuleConfig(
        rows=tuned_int("PTO_RECURRENT_GATED_DELTA_RULE_ROWS", 32, valid_values=(32, 64, 128)),
        dim=tuned_int("PTO_RECURRENT_GATED_DELTA_RULE_DIM", 16, valid_values=(16, 64, 128)),
        block_dim=tuned_int(
            "PTO_RECURRENT_GATED_DELTA_RULE_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)
        ),
    )


def _cube_meta_data(config: RecurrentGatedDeltaRuleConfig):
    bf16 = pto.bfloat16
    fp32 = pto.float32

    ptr_bf16 = pto.PtrType(bf16)
    ptr_fp32 = pto.PtrType(fp32)
    tensor_bf16 = pto.TensorType(rank=2, dtype=bf16)
    tensor_fp32 = pto.TensorType(rank=2, dtype=fp32)

    state_view = pto.SubTensorType(shape=[config.dim, config.dim], dtype=bf16)
    row_view_bf16 = pto.SubTensorType(shape=[1, config.dim], dtype=bf16)
    row_view_fp32 = pto.SubTensorType(shape=[1, config.dim], dtype=fp32)

    state_mat = pto.TileType(shape=[config.dim, config.dim], dtype=bf16, memory_space="MAT")
    state_left = pto.TileType(shape=[config.dim, config.dim], dtype=bf16, memory_space="LEFT")
    vec_mat = pto.TileType(
        shape=[16, config.dim],
        valid_shape=[1, -1],
        dtype=bf16,
        memory_space="MAT",
    )
    vec_right = pto.TileType(
        shape=[16, config.dim],
        valid_shape=[1, -1],
        dtype=bf16,
        memory_space="RIGHT",
    )
    out_acc = pto.TileType(shape=[1, config.dim], dtype=fp32, memory_space="ACC")

    return {
        "ptr_bf16": ptr_bf16,
        "ptr_fp32": ptr_fp32,
        "tensor_bf16": tensor_bf16,
        "tensor_fp32": tensor_fp32,
        "state_view": state_view,
        "row_view_bf16": row_view_bf16,
        "row_view_fp32": row_view_fp32,
        "state_mat": state_mat,
        "state_left": state_left,
        "vec_mat": vec_mat,
        "vec_right": vec_right,
        "out_acc": out_acc,
    }


def _vector_meta_data(config: RecurrentGatedDeltaRuleConfig):
    bf16 = pto.bfloat16
    fp32 = pto.float32

    ptr_bf16 = pto.PtrType(bf16)
    ptr_fp32 = pto.PtrType(fp32)
    tensor_bf16 = pto.TensorType(rank=2, dtype=bf16)
    tensor_fp32 = pto.TensorType(rank=2, dtype=fp32)

    row_view_bf16 = pto.SubTensorType(shape=[1, config.dim], dtype=bf16)
    row_view_fp32 = pto.SubTensorType(shape=[1, config.dim], dtype=fp32)
    scalar_view_bf16 = pto.SubTensorType(shape=[1, 1], dtype=bf16)
    scalar_view_fp32 = pto.SubTensorType(shape=[1, 1], dtype=fp32)

    cfg = pto.TileConfig()
    row_tile_bf16 = pto.TileType(
        shape=[1, config.dim],
        valid_shape=[1, -1],
        dtype=bf16,
        memory_space="VEC",
        config=cfg,
    )
    row_tile_fp32 = pto.TileType(
        shape=[1, config.dim],
        valid_shape=[1, -1],
        dtype=fp32,
        memory_space="VEC",
        config=cfg,
    )
    return {
        "ptr_bf16": ptr_bf16,
        "ptr_fp32": ptr_fp32,
        "tensor_bf16": tensor_bf16,
        "tensor_fp32": tensor_fp32,
        "row_view_bf16": row_view_bf16,
        "row_view_fp32": row_view_fp32,
        "scalar_view_bf16": scalar_view_bf16,
        "scalar_view_fp32": scalar_view_fp32,
        "row_tile_bf16": row_tile_bf16,
        "row_tile_fp32": row_tile_fp32,
    }


def _build_proj_stage(*, config: RecurrentGatedDeltaRuleConfig, output_dir):
    @jit(
        meta_data=lambda: _cube_meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.rows, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def recurrent_state_key_proj(
        proj_ptr: "ptr_fp32",
        state_ptr: "ptr_bf16",
        key_ptr: "ptr_bf16",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cRows = const(config.rows)
        cD = const(config.dim)

        state_rows = cRows * cD
        tv_state = pto.as_tensor(tensor_bf16, ptr=state_ptr, shape=[state_rows, cD], strides=[cD, c1])
        tv_key = pto.as_tensor(tensor_bf16, ptr=key_ptr, shape=[cRows, cD], strides=[cD, c1])
        tv_proj = pto.as_tensor(tensor_fp32, ptr=proj_ptr, shape=[cRows, cD], strides=[cD, c1])

        with pto.section.cube():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cRows, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cRows)

            state_mat_tile = pto.alloc_tile(state_mat)
            state_left_tile = pto.alloc_tile(state_left)
            key_mat_tile = pto.alloc_tile(vec_mat, valid_col=cD)
            key_right_tile = pto.alloc_tile(vec_right, valid_col=cD)
            proj_acc_tile = pto.alloc_tile(out_acc)

            for row_idx in range(row_start, row_end, c1):
                state_base = row_idx * cD
                state_view_tile = pto.slice_view(
                    state_view,
                    source=tv_state,
                    offsets=[state_base, c0],
                    sizes=[cD, cD],
                )
                key_view_tile = pto.slice_view(
                    row_view_bf16,
                    source=tv_key,
                    offsets=[row_idx, c0],
                    sizes=[c1, cD],
                )
                proj_view_tile = pto.slice_view(
                    row_view_fp32,
                    source=tv_proj,
                    offsets=[row_idx, c0],
                    sizes=[c1, cD],
                )

                pto.load(state_view_tile, state_mat_tile)
                pto.load(key_view_tile, key_mat_tile)
                pto.mov(state_mat_tile, state_left_tile)
                pto.mov(key_mat_tile, key_right_tile)
                pto.gemv(state_left_tile, key_right_tile, proj_acc_tile)
                pto.store(proj_acc_tile, proj_view_tile)

    return recurrent_state_key_proj


def _build_state_update_stage(*, config: RecurrentGatedDeltaRuleConfig, output_dir, state_row_idx_const: int):
    @jit(
        meta_data=lambda: _vector_meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.rows, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def recurrent_state_update(
        state_out_ptr: "ptr_bf16",
        state_in_ptr: "ptr_bf16",
        proj_ptr: "ptr_fp32",
        value_ptr: "ptr_bf16",
        key_ptr: "ptr_bf16",
        beta_ptr: "ptr_bf16",
        g_ptr: "ptr_fp32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cRows = const(config.rows)
        cD = const(config.dim)

        state_rows = cRows * cD
        tv_state_in = pto.as_tensor(tensor_bf16, ptr=state_in_ptr, shape=[state_rows, cD], strides=[cD, c1])
        tv_state_out = pto.as_tensor(tensor_bf16, ptr=state_out_ptr, shape=[state_rows, cD], strides=[cD, c1])
        tv_proj = pto.as_tensor(tensor_fp32, ptr=proj_ptr, shape=[cRows, cD], strides=[cD, c1])
        tv_value = pto.as_tensor(tensor_bf16, ptr=value_ptr, shape=[cRows, cD], strides=[cD, c1])
        tv_key = pto.as_tensor(tensor_bf16, ptr=key_ptr, shape=[cRows, cD], strides=[cD, c1])
        tv_beta = pto.as_tensor(tensor_bf16, ptr=beta_ptr, shape=[cRows, c1], strides=[c1, c1])
        tv_g = pto.as_tensor(tensor_fp32, ptr=g_ptr, shape=[cRows, c1], strides=[c1, c1])

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cRows, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cRows)

            value_row_bf16 = pto.alloc_tile(row_tile_bf16, valid_col=cD)
            key_row_bf16 = pto.alloc_tile(row_tile_bf16, valid_col=cD)
            state_row_bf16 = pto.alloc_tile(row_tile_bf16, valid_col=cD)
            state_new_bf16 = pto.alloc_tile(row_tile_bf16, valid_col=cD)

            value_row_fp32 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            key_row_fp32 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            proj_row_fp32 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            scaled_proj_fp32 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            delta_row_fp32 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            state_row_fp32 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            state_scaled_fp32 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            outer_row_fp32 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            alpha_row_fp32 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            beta_row_fp32 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            delta_scalar_row_fp32 = pto.alloc_tile(row_tile_fp32, valid_col=cD)

            beta_scalar_bf16 = pto.alloc_tile(row_tile_bf16, valid_col=c1)
            beta_scalar_fp32 = pto.alloc_tile(row_tile_fp32, valid_col=c1)
            alpha_scalar_fp32 = pto.alloc_tile(row_tile_fp32, valid_col=c1)
            delta_scalar_fp32 = pto.alloc_tile(row_tile_fp32, valid_col=c1)

            for row_idx in range(row_start, row_end, c1):
                value_view = pto.slice_view(row_view_bf16, source=tv_value, offsets=[row_idx, c0], sizes=[c1, cD])
                key_view = pto.slice_view(row_view_bf16, source=tv_key, offsets=[row_idx, c0], sizes=[c1, cD])
                proj_view = pto.slice_view(row_view_fp32, source=tv_proj, offsets=[row_idx, c0], sizes=[c1, cD])
                beta_view = pto.slice_view(scalar_view_bf16, source=tv_beta, offsets=[row_idx, c0], sizes=[c1, c1])
                g_view = pto.slice_view(scalar_view_fp32, source=tv_g, offsets=[row_idx, c0], sizes=[c1, c1])

                pto.load(value_view, value_row_bf16)
                pto.load(key_view, key_row_bf16)
                pto.load(proj_view, proj_row_fp32)
                pto.load(beta_view, beta_scalar_bf16)
                pto.load(g_view, alpha_scalar_fp32)

                pto.cvt(value_row_bf16, value_row_fp32)
                pto.cvt(key_row_bf16, key_row_fp32)
                pto.cvt(beta_scalar_bf16, beta_scalar_fp32)
                pto.exp(alpha_scalar_fp32, alpha_scalar_fp32)
                pto.row_expand(alpha_scalar_fp32, alpha_row_fp32)
                pto.row_expand(beta_scalar_fp32, beta_row_fp32)
                pto.mul(proj_row_fp32, alpha_row_fp32, scaled_proj_fp32)
                pto.sub(value_row_fp32, scaled_proj_fp32, delta_row_fp32)
                state_row_idx = const(state_row_idx_const)
                state_base = row_idx * cD + state_row_idx
                state_view = pto.slice_view(
                    row_view_bf16,
                    source=tv_state_in,
                    offsets=[state_base, c0],
                    sizes=[c1, cD],
                )
                state_out_view = pto.slice_view(
                    row_view_bf16,
                    source=tv_state_out,
                    offsets=[state_base, c0],
                    sizes=[c1, cD],
                )

                pto.load(state_view, state_row_bf16)
                pto.cvt(state_row_bf16, state_row_fp32)
                pto.extract(delta_row_fp32, c0, state_row_idx, delta_scalar_fp32)
                pto.row_expand(delta_scalar_fp32, delta_scalar_row_fp32)
                pto.mul(delta_scalar_row_fp32, key_row_fp32, outer_row_fp32)
                pto.mul(outer_row_fp32, beta_row_fp32, outer_row_fp32)
                pto.mul(state_row_fp32, alpha_row_fp32, state_scaled_fp32)
                pto.add(state_scaled_fp32, outer_row_fp32, state_scaled_fp32)
                pto.cvt(state_scaled_fp32, state_new_bf16)
                pto.store(state_new_bf16, state_out_view)

    return recurrent_state_update


def _build_out_stage(*, config: RecurrentGatedDeltaRuleConfig, output_dir):
    @jit(
        meta_data=lambda: _cube_meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.rows, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def recurrent_state_query_proj(
        out_tmp_ptr: "ptr_fp32",
        state_ptr: "ptr_bf16",
        query_ptr: "ptr_bf16",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cRows = const(config.rows)
        cD = const(config.dim)

        state_rows = cRows * cD
        tv_state = pto.as_tensor(tensor_bf16, ptr=state_ptr, shape=[state_rows, cD], strides=[cD, c1])
        tv_query = pto.as_tensor(tensor_bf16, ptr=query_ptr, shape=[cRows, cD], strides=[cD, c1])
        tv_out_tmp = pto.as_tensor(tensor_fp32, ptr=out_tmp_ptr, shape=[cRows, cD], strides=[cD, c1])

        with pto.section.cube():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cRows, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cRows)

            state_mat_tile = pto.alloc_tile(state_mat)
            state_left_tile = pto.alloc_tile(state_left)
            query_mat_tile = pto.alloc_tile(vec_mat, valid_col=cD)
            query_right_tile = pto.alloc_tile(vec_right, valid_col=cD)
            out_acc_tile = pto.alloc_tile(out_acc)

            for row_idx in range(row_start, row_end, c1):
                state_base = row_idx * cD
                state_view_tile = pto.slice_view(
                    state_view,
                    source=tv_state,
                    offsets=[state_base, c0],
                    sizes=[cD, cD],
                )
                query_view_tile = pto.slice_view(
                    row_view_bf16,
                    source=tv_query,
                    offsets=[row_idx, c0],
                    sizes=[c1, cD],
                )
                out_view_tile = pto.slice_view(
                    row_view_fp32,
                    source=tv_out_tmp,
                    offsets=[row_idx, c0],
                    sizes=[c1, cD],
                )

                pto.load(state_view_tile, state_mat_tile)
                pto.load(query_view_tile, query_mat_tile)
                pto.mov(state_mat_tile, state_left_tile)
                pto.mov(query_mat_tile, query_right_tile)
                pto.gemv(state_left_tile, query_right_tile, out_acc_tile)
                pto.store(out_acc_tile, out_view_tile)

    return recurrent_state_query_proj


def _build_scale_stage(*, config: RecurrentGatedDeltaRuleConfig, output_dir):
    scale = 1.0 / (float(config.dim) ** 0.5)

    @jit(
        meta_data=lambda: _vector_meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.rows, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def recurrent_scale_out(
        out_ptr: "ptr_bf16",
        out_tmp_ptr: "ptr_fp32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cRows = const(config.rows)
        cD = const(config.dim)
        scale_const = const(scale, dtype=pto.float32)

        tv_out = pto.as_tensor(tensor_bf16, ptr=out_ptr, shape=[cRows, cD], strides=[cD, c1])
        tv_out_tmp = pto.as_tensor(tensor_fp32, ptr=out_tmp_ptr, shape=[cRows, cD], strides=[cD, c1])

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cRows, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cRows)

            out_row_fp32 = pto.alloc_tile(row_tile_fp32, valid_col=cD)
            out_row_bf16 = pto.alloc_tile(row_tile_bf16, valid_col=cD)

            for row_idx in range(row_start, row_end, c1):
                out_tmp_view = pto.slice_view(row_view_fp32, source=tv_out_tmp, offsets=[row_idx, c0], sizes=[c1, cD])
                out_view = pto.slice_view(row_view_bf16, source=tv_out, offsets=[row_idx, c0], sizes=[c1, cD])

                pto.load(out_tmp_view, out_row_fp32)
                pto.muls(out_row_fp32, scale_const, out_row_fp32)
                pto.cvt(out_row_fp32, out_row_bf16)
                pto.store(out_row_bf16, out_view)

    return recurrent_scale_out


class RecurrentGatedDeltaRuleWrapper:
    def __init__(self, *, output_dir):
        self._output_dir = Path(output_dir)
        self._config = _config()
        self._proj_stage = _build_proj_stage(config=self._config, output_dir=self._output_dir / "stage_proj")
        self._state_stages = tuple(
            _build_state_update_stage(
                config=self._config,
                output_dir=self._output_dir / f"stage_state_update_row_{state_row_idx:03d}",
                state_row_idx_const=state_row_idx,
            )
            for state_row_idx in range(self._config.dim)
        )
        self._out_stage = _build_out_stage(config=self._config, output_dir=self._output_dir / "stage_out")
        self._scale_stage = _build_scale_stage(config=self._config, output_dir=self._output_dir / "stage_scale")

    def _build(self):
        for stage in (self._proj_stage, *self._state_stages, self._out_stage, self._scale_stage):
            stage._build()

    def _artifact_paths(self):
        paths = []
        for stage in (self._proj_stage, *self._state_stages, self._out_stage, self._scale_stage):
            paths.extend(stage._artifact_paths())
        return tuple(paths)

    @property
    def output_dir(self):
        return str(self._output_dir)

    def __call__(
        self,
        out,
        final_state,
        query,
        key,
        value,
        state,
        beta,
        g,
        stream_ptr=None,
    ):
        device = query.device
        proj_tmp = torch.empty((self._config.rows, self._config.dim), device=device, dtype=torch.float32)
        out_tmp = torch.empty((self._config.rows, self._config.dim), device=device, dtype=torch.float32)

        self._proj_stage(proj_tmp, state, key, stream_ptr=stream_ptr)
        current_state = state
        for stage_idx, state_stage in enumerate(self._state_stages):
            state_stage(final_state, current_state, proj_tmp, value, key, beta, g, stream_ptr=stream_ptr)
            if stage_idx + 1 < len(self._state_stages):
                current_state = final_state
        self._out_stage(out_tmp, final_state, query, stream_ptr=stream_ptr)
        self._scale_stage(out, out_tmp, stream_ptr=stream_ptr)


def build_jit_wrapper(*, output_dir):
    return RecurrentGatedDeltaRuleWrapper(output_dir=output_dir)
