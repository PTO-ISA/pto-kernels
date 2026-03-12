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


CONFIG = DenseReluFfnConfig(
    tokens=32,
    hidden=128,
    intermediate=256,
    output=128,
    base_k1=32,
    base_k2=32,
)


def build_jit_wrapper(*, output_dir):
    return DenseReluFfnPipelineWrapper(config=CONFIG, output_dir=output_dir)
