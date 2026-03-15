"""Constrained PTO-DSL seed for moe_gating_top_k_softmax on 910B."""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class MoeGatingTopKSoftmaxConfig:
    rows: int
    experts: int
    block_dim: int


def _config() -> MoeGatingTopKSoftmaxConfig:
    return MoeGatingTopKSoftmaxConfig(
        rows=tuned_int("PTO_MOE_GATING_TOPK_SOFTMAX_ROWS", 8, valid_values=(8, 128, 256)),
        experts=tuned_int("PTO_MOE_GATING_TOPK_SOFTMAX_EXPERTS", 16, valid_values=(16, 64, 128)),
        block_dim=tuned_int("PTO_MOE_GATING_TOPK_SOFTMAX_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)),
    )


def _softmax_meta_data(config: MoeGatingTopKSoftmaxConfig):
    dtype = pto.float16
    ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)
    row_view = pto.SubTensorType(shape=[1, config.experts], dtype=dtype)
    cfg = pto.TileConfig()
    row_tile = pto.TileType(
        shape=[1, config.experts],
        valid_shape=[1, -1],
        dtype=dtype,
        memory_space="VEC",
        config=cfg,
    )
    return {
        "ptr": ptr,
        "tensor": tensor,
        "row_view": row_view,
        "row_tile": row_tile,
    }


def _select_meta_data(config: MoeGatingTopKSoftmaxConfig):
    dtype = pto.float16
    i32 = pto.int32
    ptr = pto.PtrType(dtype)
    ptr_i32 = pto.PtrType(i32)
    return {
        "ptr": ptr,
        "ptr_i32": ptr_i32,
    }


def _build_softmax_stage(*, config: MoeGatingTopKSoftmaxConfig, output_dir):
    @jit(
        meta_data=lambda: _softmax_meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.rows, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_gating_top_k_softmax_stage(probs_ptr: "ptr", x_ptr: "ptr") -> None:
        c0 = const(0)
        c1 = const(1)
        cRows = const(config.rows)
        cExperts = const(config.experts)

        tv_x = pto.as_tensor(tensor, ptr=x_ptr, shape=[cRows, cExperts], strides=[cExperts, c1])
        tv_probs = pto.as_tensor(tensor, ptr=probs_ptr, shape=[cRows, cExperts], strides=[cExperts, c1])

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cRows, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cRows)

            row_in = pto.alloc_tile(row_tile, valid_col=cExperts)
            row_tmp = pto.alloc_tile(row_tile, valid_col=cExperts)
            row_tmp2 = pto.alloc_tile(row_tile, valid_col=cExperts)
            scalar = pto.alloc_tile(row_tile, valid_col=c1)
            scalar_expanded = pto.alloc_tile(row_tile, valid_col=cExperts)

            for row_idx in range(row_start, row_end, c1):
                x_view = pto.slice_view(row_view, source=tv_x, offsets=[row_idx, c0], sizes=[c1, cExperts])
                out_view = pto.slice_view(row_view, source=tv_probs, offsets=[row_idx, c0], sizes=[c1, cExperts])
                pto.load(x_view, row_in)
                pto.row_max(row_in, row_tmp, scalar)
                pto.row_expand(scalar, scalar_expanded)
                pto.sub(row_in, scalar_expanded, row_tmp)
                pto.exp(row_tmp, row_tmp)
                pto.row_sum(row_tmp, row_tmp2, scalar)
                pto.row_expand(scalar, scalar_expanded)
                pto.div(row_tmp, scalar_expanded, row_tmp2)
                pto.store(row_tmp2, out_view)

    return moe_gating_top_k_softmax_stage


def _build_select_stage(*, config: MoeGatingTopKSoftmaxConfig, output_dir):
    @jit(
        meta_data=lambda: _select_meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.rows, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_gating_top_k_select_stage(
        y_ptr: "ptr",
        expert_idx_ptr: "ptr_i32",
        row_idx_ptr: "ptr_i32",
        probs_ptr: "ptr",
        x_ptr: "ptr",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cRows = const(config.rows)
        cExperts = const(config.experts)

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cRows, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cRows)

            for row_idx in range(row_start, row_end, c1):
                row_off = row_idx * cExperts
                best_idx_i32 = pto.index_cast(c0, pto.int32)
                best_val = pto.load_scalar(pto.float16, x_ptr, row_off + c0)
                for expert_idx_int in range(1, config.experts):
                    expert_idx = expert_idx_int
                    expert_idx_i32 = pto.index_cast(expert_idx, pto.int32)
                    current = pto.load_scalar(pto.float16, x_ptr, row_off + expert_idx)
                    take_current = pto.gt(current, best_val)
                    best_val = pto.select(take_current, current, best_val)
                    best_idx_i32 = pto.select(take_current, expert_idx_i32, best_idx_i32)

                best_idx = pto.index_cast(best_idx_i32)
                selected_prob = pto.load_scalar(pto.float16, probs_ptr, row_off + best_idx)
                pto.store_scalar(y_ptr, row_idx, selected_prob)
                pto.store_scalar(expert_idx_ptr, row_idx, best_idx_i32)
                pto.store_scalar(row_idx_ptr, row_idx, pto.index_cast(row_idx, pto.int32))

    return moe_gating_top_k_select_stage


class MoeGatingTopKSoftmaxWrapper:
    def __init__(self, *, output_dir):
        self._output_dir = Path(output_dir)
        self._config = _config()
        self._softmax = _build_softmax_stage(
            config=self._config,
            output_dir=self._output_dir / "stage1_softmax",
        )
        self._select = _build_select_stage(
            config=self._config,
            output_dir=self._output_dir / "stage2_select",
        )

    def _build(self):
        self._softmax._build()
        self._select._build()

    def _artifact_paths(self):
        return (
            *self._softmax._artifact_paths(),
            *self._select._artifact_paths(),
        )

    @property
    def library_path(self):
        return None

    @property
    def output_dir(self):
        return str(self._output_dir)

    def __call__(self, y_ptr, expert_idx_ptr, row_idx_ptr, x_ptr, probs_ptr, stream_ptr=None):
        self._softmax(probs_ptr, x_ptr, stream_ptr=stream_ptr)
        self._select(y_ptr, expert_idx_ptr, row_idx_ptr, probs_ptr, x_ptr, stream_ptr=stream_ptr)


def build_jit_wrapper(*, output_dir):
    return MoeGatingTopKSoftmaxWrapper(output_dir=output_dir)
