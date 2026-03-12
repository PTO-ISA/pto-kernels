"""Constrained PTO-DSL seed for flash_attention_score on 910B.

The seed remains intentionally staged:
1. dense QK matmul
2. dense row-wise stable softmax
3. dense PV matmul

The staged builders now live in `pto_kernels.ops.attention.common` so later
attention-family ports can reuse the same dense baseline path while masked,
online, and sparse variants are brought up separately.
"""

from pto_kernels.ops.attention.common import DenseAttentionConfig, DenseAttentionPipelineWrapper


CONFIG = DenseAttentionConfig(
    seq_len=32,
    head_dim=64,
    scores_dim=32,
    qk_base_k=32,
    pv_base_k=32,
)


def build_jit_wrapper(*, output_dir):
    return DenseAttentionPipelineWrapper(config=CONFIG, output_dir=output_dir)
