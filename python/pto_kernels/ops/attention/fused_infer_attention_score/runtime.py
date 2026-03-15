"""Runtime helpers for the first fused_infer_attention_score migration slice."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class FusedInferAttentionScoreVariant:
    q_seq: int
    kv_seq: int
    head_dim: int
    seed: int
    batch_size: int = 1
    q_heads: int = 1
    kv_heads: int = 1
    input_layout: str = "BNSD"
    dtype: str = "float16"

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"q{self.q_seq}_k{self.kv_seq}_d{self.head_dim}"

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def block_size(self) -> int:
        return self.kv_seq

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "query": [self.batch_size, self.q_heads, self.q_seq, self.head_dim],
            "k_cache": [1, self.block_size, self.kv_heads * self.head_dim],
            "v_cache": [1, self.block_size, self.kv_heads * self.head_dim],
            "block_table": [self.batch_size, 1],
            "attention_out": [self.batch_size, self.q_heads, self.q_seq, self.head_dim],
        }


VARIANT = FusedInferAttentionScoreVariant(q_seq=16, kv_seq=16, head_dim=16, seed=0)
VARIANTS = (
    FusedInferAttentionScoreVariant(q_seq=16, kv_seq=16, head_dim=16, seed=0),
    FusedInferAttentionScoreVariant(q_seq=64, kv_seq=64, head_dim=64, seed=1),
    FusedInferAttentionScoreVariant(q_seq=32, kv_seq=128, head_dim=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _reference_cpu(query: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, scale: float) -> torch.Tensor:
    q = query[0, 0].float()
    k = k_cache[0].view(-1, q.shape[-1]).float()
    v = v_cache[0].view(-1, q.shape[-1]).float()
    scores = torch.matmul(q, k.transpose(0, 1)) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v).to(torch.float16).float()
    return out.view(1, 1, q.shape[0], q.shape[1])


def make_fused_infer_attention_score_inputs(
    variant: FusedInferAttentionScoreVariant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    g = torch.Generator(device="cpu")
    g.manual_seed(variant.seed)

    query_cpu = torch.randn(
        (variant.batch_size, variant.q_heads, variant.q_seq, variant.head_dim),
        generator=g,
        dtype=torch.float32,
    ).to(torch.float16)
    k_cache_cpu = torch.randn(
        (1, variant.block_size, variant.kv_heads * variant.head_dim),
        generator=g,
        dtype=torch.float32,
    ).to(torch.float16)
    v_cache_cpu = torch.randn(
        (1, variant.block_size, variant.kv_heads * variant.head_dim),
        generator=g,
        dtype=torch.float32,
    ).to(torch.float16)
    block_table_cpu = torch.zeros((variant.batch_size, 1), dtype=torch.int32)

    reference = _reference_cpu(query_cpu, k_cache_cpu, v_cache_cpu, variant.scale)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "query": query_cpu.npu(),
        "query_pto": query_cpu.mul(variant.scale).to(torch.float16).npu(),
        "k_cache": k_cache_cpu.npu(),
        "v_cache": v_cache_cpu.npu(),
        "block_table": block_table_cpu.npu(),
        "out_pto": torch.empty_like(query_cpu).npu(),
        "scores_pto": torch.empty((variant.q_seq, variant.kv_seq), dtype=torch.float16).npu(),
        "query_flat": query_cpu[0, 0].contiguous().npu(),
        "key_flat": k_cache_cpu[0].view(variant.kv_seq, variant.head_dim).contiguous().npu(),
        "key_t_flat": k_cache_cpu[0].view(variant.kv_seq, variant.head_dim).transpose(0, 1).contiguous().npu(),
        "value_flat": v_cache_cpu[0].view(variant.kv_seq, variant.head_dim).contiguous().npu(),
        "reference": reference,
    }


def run_torch_npu_fused_infer_attention_score(inputs: dict[str, object]):
    variant = inputs["variant"]
    return torch_npu.npu_fused_infer_attention_score(
        inputs["query"],
        inputs["k_cache"],
        inputs["v_cache"],
        actual_seq_lengths=[variant["q_seq"]],
        actual_seq_lengths_kv=[variant["kv_seq"]],
        num_heads=variant["q_heads"],
        num_key_value_heads=variant["kv_heads"],
        input_layout=variant["input_layout"],
        scale=1.0 / math.sqrt(variant["head_dim"]),
        pre_tokens=65535,
        next_tokens=65535,
        block_table=inputs["block_table"],
        block_size=variant["kv_seq"],
    )


def run_pto_fused_infer_attention_score_variant(wrapper, inputs: dict[str, object]) -> torch.Tensor:
    wrapper(
        inputs["out_pto"],
        inputs["scores_pto"],
        inputs["query_pto"][0, 0].contiguous(),
        inputs["key_t_flat"],
        inputs["value_flat"],
    )
    return inputs["out_pto"].float()
