"""Constrained PTO-DSL seed for prompt_flash_attention on 910B/A3."""

from dataclasses import dataclass
from pathlib import Path

from pto_kernels.ops.attention.common import DenseAttentionConfig, DenseAttentionPipelineWrapper
from pto_kernels.utils.tuning import tuned_int


@dataclass(frozen=True)
class PromptFlashAttentionConfig:
    q_seq: int
    kv_seq: int
    head_dim: int
    qk_base_m: int
    qk_base_n: int
    qk_base_k: int
    qk_block_dim: int
    pv_base_m: int
    pv_base_n: int
    pv_base_k: int
    pv_block_dim: int
    softmax_block_dim: int


def _config() -> PromptFlashAttentionConfig:
    return PromptFlashAttentionConfig(
        q_seq=tuned_int("PTO_PROMPT_FLASH_ATTENTION_Q_SEQ", 16, valid_values=(16, 32, 64)),
        kv_seq=tuned_int("PTO_PROMPT_FLASH_ATTENTION_KV_SEQ", 16, valid_values=(16, 64, 128)),
        head_dim=tuned_int("PTO_PROMPT_FLASH_ATTENTION_HEAD_DIM", 16, valid_values=(16, 64, 128)),
        qk_base_m=tuned_int("PTO_PROMPT_FLASH_ATTENTION_QK_BASE_M", 16, valid_values=(16, 32, 64)),
        qk_base_n=tuned_int("PTO_PROMPT_FLASH_ATTENTION_QK_BASE_N", 16, valid_values=(16, 32, 64)),
        qk_base_k=tuned_int("PTO_PROMPT_FLASH_ATTENTION_QK_BASE_K", 16, valid_values=(16, 32, 64)),
        qk_block_dim=tuned_int("PTO_PROMPT_FLASH_ATTENTION_QK_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)),
        pv_base_m=tuned_int("PTO_PROMPT_FLASH_ATTENTION_PV_BASE_M", 16, valid_values=(16, 32, 64)),
        pv_base_n=tuned_int("PTO_PROMPT_FLASH_ATTENTION_PV_BASE_N", 16, valid_values=(16, 32, 64, 128)),
        pv_base_k=tuned_int("PTO_PROMPT_FLASH_ATTENTION_PV_BASE_K", 16, valid_values=(16, 32, 64, 128)),
        pv_block_dim=tuned_int("PTO_PROMPT_FLASH_ATTENTION_PV_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)),
        softmax_block_dim=tuned_int("PTO_PROMPT_FLASH_ATTENTION_SOFTMAX_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16, 20)),
    )


class PromptFlashAttentionWrapper:
    def __init__(self, *, output_dir):
        self._output_dir = Path(output_dir)
        cfg = _config()
        self._config = cfg
        self._pipeline = DenseAttentionPipelineWrapper(
            config=DenseAttentionConfig(
                seq_len=cfg.q_seq,
                head_dim=cfg.head_dim,
                scores_dim=cfg.kv_seq,
                qk_base_m=cfg.qk_base_m,
                qk_base_n=cfg.qk_base_n,
                qk_base_k=cfg.qk_base_k,
                qk_block_dim=cfg.qk_block_dim,
                pv_base_m=cfg.pv_base_m,
                pv_base_n=cfg.pv_base_n,
                pv_base_k=cfg.pv_base_k,
                pv_block_dim=cfg.pv_block_dim,
                softmax_block_dim=cfg.softmax_block_dim,
                softmax_fp32=True,
            ),
            output_dir=self._output_dir,
        )

    def _build(self):
        self._pipeline._build()

    def _artifact_paths(self):
        return tuple(self._pipeline._artifact_paths())

    @property
    def output_dir(self):
        return str(self._output_dir)

    def __call__(self, out, scores, query, key_t, value, stream_ptr=None):
        self._pipeline(out, scores, query, key_t, value, stream_ptr=stream_ptr)


def build_jit_wrapper(*, output_dir):
    return PromptFlashAttentionWrapper(output_dir=output_dir)
