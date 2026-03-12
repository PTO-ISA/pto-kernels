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
        qk_base_m=tuned_int("PTO_ATTENTION_QK_BASE_M", 16, valid_values=(16, 32, 64)),
        qk_base_n=tuned_int("PTO_ATTENTION_QK_BASE_N", 16, valid_values=(16, 32, 64)),
        qk_base_k=tuned_int("PTO_ATTENTION_QK_BASE_K", 64, valid_values=(32, 64)),
        qk_block_dim=tuned_int("PTO_ATTENTION_QK_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)),
        pv_base_m=tuned_int("PTO_ATTENTION_PV_BASE_M", 16, valid_values=(16, 32, 64)),
        pv_base_n=tuned_int("PTO_ATTENTION_PV_BASE_N", 32, valid_values=(32, 64, 128)),
        pv_base_k=tuned_int("PTO_ATTENTION_PV_BASE_K", 32, valid_values=(16, 32, 64)),
        pv_block_dim=tuned_int("PTO_ATTENTION_PV_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)),
        softmax_block_dim=tuned_int("PTO_ATTENTION_SOFTMAX_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)),
    )


def build_jit_wrapper(*, output_dir):
    return DenseAttentionPipelineWrapper(config=_config(), output_dir=output_dir)
