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
from pto_kernels.utils.tuning import tuned_int


def _config() -> DenseAttentionConfig:
    seq_len = tuned_int("PTO_ATTENTION_SEQ_LEN", 32, valid_values=(32, 64))
    return DenseAttentionConfig(
        seq_len=seq_len,
        head_dim=tuned_int("PTO_ATTENTION_HEAD_DIM", 64, valid_values=(64, 128)),
        scores_dim=seq_len,
        qk_base_k=tuned_int("PTO_ATTENTION_QK_BASE_K", 64, valid_values=(32, 64)),
        pv_base_k=32,
    )


def build_jit_wrapper(*, output_dir):
    return DenseAttentionPipelineWrapper(config=_config(), output_dir=output_dir)
