"""Constrained PTO-DSL seed for moe_gating_top_k on 910B."""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class MoeGatingTopKConfig:
    rows: int
    experts: int
    block_dim: int


def _config() -> MoeGatingTopKConfig:
    return MoeGatingTopKConfig(
        rows=tuned_int("PTO_MOE_GATING_TOPK_ROWS", 8, valid_values=(8, 128, 256)),
        experts=tuned_int("PTO_MOE_GATING_TOPK_EXPERTS", 16, valid_values=(16, 64, 128)),
        block_dim=tuned_int("PTO_MOE_GATING_TOPK_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)),
    )


def _select_meta_data(config: MoeGatingTopKConfig):
    dtype = pto.float16
    i32 = pto.int32
    return {
        "ptr": pto.PtrType(dtype),
        "ptr_i32": pto.PtrType(i32),
    }


def _build_select_stage(*, config: MoeGatingTopKConfig, output_dir):
    @jit(
        meta_data=lambda: _select_meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.rows, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_gating_top_k_select_stage(y_ptr: "ptr", expert_idx_ptr: "ptr_i32", x_ptr: "ptr") -> None:
        c0 = const(0)
        c1 = const(1)
        cRows = const(config.rows)
        cExperts = const(config.experts)
        one_h = const(1.0, dtype=pto.float16)

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

                pto.store_scalar(y_ptr, row_idx, one_h)
                pto.store_scalar(expert_idx_ptr, row_idx, best_idx_i32)

    return moe_gating_top_k_select_stage


class MoeGatingTopKWrapper:
    def __init__(self, *, output_dir):
        self._output_dir = Path(output_dir)
        self._config = _config()
        self._select = _build_select_stage(
            config=self._config,
            output_dir=self._output_dir / "stage_select",
        )

    def _build(self):
        self._select._build()

    def _artifact_paths(self):
        return tuple(self._select._artifact_paths())

    @property
    def library_path(self):
        return None

    @property
    def output_dir(self):
        return str(self._output_dir)

    def __call__(self, y_ptr, expert_idx_ptr, x_ptr, stream_ptr=None):
        self._select(y_ptr, expert_idx_ptr, x_ptr, stream_ptr=stream_ptr)


def build_jit_wrapper(*, output_dir):
    return MoeGatingTopKWrapper(output_dir=output_dir)
