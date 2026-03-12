"""Constrained PTO-DSL seed for dense relu FFN on 910B.

The phase-1 FFN path remains a staged pipeline:
1. dense matmul `x @ w1 -> hidden`
2. vector relu on `hidden`
3. dense matmul `hidden @ w2 -> out`

The reusable staged builder now lives in `pto_kernels.ops.ffn.common` so later
FFN-family kernels can share the same bring-up contract before fused lowering
lands in PTODSL/PTOAS.
"""

from pto_kernels.ops.ffn.common import DenseReluFfnConfig, DenseReluFfnPipelineWrapper
from pto_kernels.utils.tuning import tuned_int


def _config() -> DenseReluFfnConfig:
    return DenseReluFfnConfig(
        tokens=tuned_int("PTO_FFN_TOKENS", 32, valid_values=(32, 64)),
        hidden=tuned_int("PTO_FFN_HIDDEN", 128, valid_values=(128,)),
        intermediate=tuned_int("PTO_FFN_INTERMEDIATE", 256, valid_values=(256, 512)),
        output=tuned_int("PTO_FFN_OUTPUT", 128, valid_values=(128,)),
        base_m1=tuned_int("PTO_FFN_BASE_M1", 16, valid_values=(16, 32, 64)),
        base_n1=tuned_int("PTO_FFN_BASE_N1", 64, valid_values=(64, 128, 256)),
        base_k1=tuned_int("PTO_FFN_BASE_K1", 32, valid_values=(32, 64)),
        block_dim1=tuned_int("PTO_FFN_BLOCK_DIM1", 8, valid_values=(1, 2, 4, 8, 16, 20)),
        base_m2=tuned_int("PTO_FFN_BASE_M2", 16, valid_values=(16, 32, 64)),
        base_n2=tuned_int("PTO_FFN_BASE_N2", 64, valid_values=(64, 128)),
        base_k2=tuned_int("PTO_FFN_BASE_K2", 64, valid_values=(32, 64)),
        block_dim2=tuned_int("PTO_FFN_BLOCK_DIM2", 8, valid_values=(1, 2, 4, 8, 16, 20)),
        relu_block_dim=tuned_int("PTO_FFN_RELU_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)),
    )


def build_jit_wrapper(*, output_dir):
    return DenseReluFfnPipelineWrapper(config=_config(), output_dir=output_dir)
