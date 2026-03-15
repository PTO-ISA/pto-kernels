"""Runtime helpers for the first incre_flash_attention migration slice."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class IncreFlashAttentionVariant:
    kv_seq: int
    head_dim: int
    seed: int
    batch_size: int = 1
    q_seq: int = 1
    q_heads: int = 16
    kv_heads: int = 1
    input_layout: str = "BNSD"
    dtype: str = "float16"
    input_scale: float = 0.125

    def as_dict(self) -> dict[str, int | str | float]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"h{self.q_heads}_k{self.kv_seq}_d{self.head_dim}"

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "query": [self.batch_size, self.q_heads, self.q_seq, self.head_dim],
            "key": [self.batch_size, self.kv_heads, self.kv_seq, self.head_dim],
            "value": [self.batch_size, self.kv_heads, self.kv_seq, self.head_dim],
            "attention_out": [self.batch_size, self.q_heads, self.q_seq, self.head_dim],
        }


VARIANT = IncreFlashAttentionVariant(kv_seq=16, head_dim=16, seed=0)
VARIANTS = (
    IncreFlashAttentionVariant(kv_seq=16, head_dim=16, seed=0),
    IncreFlashAttentionVariant(kv_seq=128, head_dim=64, seed=1),
    IncreFlashAttentionVariant(kv_seq=128, head_dim=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _reference_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    q = query[0, :, 0, :].float()
    k = key[0, 0].float()
    v = value[0, 0].float()
    scores = torch.matmul(q, k.transpose(0, 1)) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v).to(torch.float16).float()
    return out.view(1, q.shape[0], 1, q.shape[1])


def make_incre_flash_attention_inputs(
    variant: IncreFlashAttentionVariant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    g = torch.Generator(device="cpu")
    g.manual_seed(variant.seed)

    query_cpu = (
        torch.randn(
            (variant.batch_size, variant.q_heads, variant.q_seq, variant.head_dim),
            generator=g,
            dtype=torch.float32,
        )
        .mul(variant.input_scale)
        .to(torch.float16)
    )
    key_cpu = (
        torch.randn(
            (variant.batch_size, variant.kv_heads, variant.kv_seq, variant.head_dim),
            generator=g,
            dtype=torch.float32,
        )
        .mul(variant.input_scale)
        .to(torch.float16)
    )
    value_cpu = (
        torch.randn(
            (variant.batch_size, variant.kv_heads, variant.kv_seq, variant.head_dim),
            generator=g,
            dtype=torch.float32,
        )
        .mul(variant.input_scale)
        .to(torch.float16)
    )

    reference = _reference_cpu(query_cpu, key_cpu, value_cpu, variant.scale)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "query": query_cpu.npu(),
        "key": key_cpu.npu(),
        "value": value_cpu.npu(),
        "query_pto": query_cpu.mul(variant.scale).to(torch.float16).npu(),
        "key_t": key_cpu[0, 0].transpose(-1, -2).contiguous().npu(),
        "scores_pto": torch.empty((variant.q_heads, variant.kv_seq), dtype=torch.float16).npu(),
        "output_pto": torch.empty_like(query_cpu).npu(),
        "reference": reference,
    }


def run_torch_npu_incre_flash_attention(inputs: dict[str, object]):
    variant = inputs["variant"]
    return torch_npu.npu_incre_flash_attention(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        actual_seq_lengths=[variant["kv_seq"]],
        num_heads=variant["q_heads"],
        scale_value=1.0 / math.sqrt(variant["head_dim"]),
        input_layout=variant["input_layout"],
        num_key_value_heads=variant["kv_heads"],
        inner_precise=1,
    )


def run_pto_incre_flash_attention_variant(wrapper, inputs: dict[str, object]) -> torch.Tensor:
    wrapper(
        inputs["output_pto"][0, :, 0, :].contiguous(),
        inputs["scores_pto"],
        inputs["query_pto"][0, :, 0, :].contiguous(),
        inputs["key_t"],
        inputs["value"][0, 0].contiguous(),
    )
    return inputs["output_pto"].float()
