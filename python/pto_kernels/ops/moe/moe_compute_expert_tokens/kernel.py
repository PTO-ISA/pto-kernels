"""Constrained PTO-DSL seed for moe_compute_expert_tokens on 910B."""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class MoeComputeExpertTokensConfig:
    rows: int
    num_experts: int
    block_dim: int


def _config() -> MoeComputeExpertTokensConfig:
    return MoeComputeExpertTokensConfig(
        rows=tuned_int("PTO_MOE_COMPUTE_EXPERT_TOKENS_ROWS", 64, valid_values=(64, 4096, 8192)),
        num_experts=tuned_int("PTO_MOE_COMPUTE_EXPERT_TOKENS_EXPERTS", 8, valid_values=(8, 64, 128)),
        block_dim=tuned_int("PTO_MOE_COMPUTE_EXPERT_TOKENS_BLOCK_DIM", 20, valid_values=(1, 2, 4, 8, 16, 20)),
    )


def _meta_data(config: MoeComputeExpertTokensConfig):
    i32 = pto.int32
    ptr_i32 = pto.PtrType(i32)
    tensor_i32 = pto.TensorType(rank=1, dtype=i32)
    return {
        "ptr_i32": ptr_i32,
        "tensor_i32": tensor_i32,
    }


def _build_kernel(*, config: MoeComputeExpertTokensConfig, output_dir):
    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.num_experts, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def moe_compute_expert_tokens_seed(out_ptr: "ptr_i32", sorted_experts_ptr: "ptr_i32") -> None:
        c0 = const(0)
        c1 = const(1)
        cRows = const(config.rows)
        cExperts = const(config.num_experts)
        i32 = pto.int32

        _ = pto.as_tensor(
            tensor_i32,
            ptr=sorted_experts_ptr,
            shape=[cRows],
            strides=[c1],
        )
        _ = pto.as_tensor(
            tensor_i32,
            ptr=out_ptr,
            shape=[cExperts],
            strides=[c1],
        )

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            experts_per_core = pto.ceil_div(cExperts, num_blocks)
            expert_start = bid * experts_per_core
            expert_end = pto.min_u(expert_start + experts_per_core, cExperts)

            for expert_idx in range(expert_start, expert_end, c1):
                target = pto.index_cast(expert_idx + c1, i32)
                expert_pos = pto.index_cast(c0, i32)
                for row_idx in range(c0, cRows, c1):
                    value = pto.load_scalar(i32, sorted_experts_ptr, row_idx)
                    is_lt = value < target
                    next_pos = pto.index_cast(row_idx + c1, i32)
                    expert_pos = pto.select(is_lt, next_pos, expert_pos)
                pto.store_scalar(out_ptr, expert_idx, expert_pos)

    return moe_compute_expert_tokens_seed


class MoeComputeExpertTokensWrapper:
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

    def __call__(self, out_ptr, sorted_experts_ptr, stream_ptr=None):
        self._kernel(out_ptr, sorted_experts_ptr, stream_ptr=stream_ptr)


def build_jit_wrapper(*, output_dir):
    return MoeComputeExpertTokensWrapper(output_dir=output_dir)
