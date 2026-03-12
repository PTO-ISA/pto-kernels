"""Runtime helpers for the first flash_attention_score migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class DenseBnsdVariant:
    batch: int = 1
    heads: int = 1
    seq_len: int = 32
    head_dim: int = 64
    seed: int = 0
    input_scale: float = 0.125
    dtype: str = "float16"
    input_layout: str = "BNSD"
    sparse_mode: int = 0
    keep_prob: float = 1.0

    def as_dict(self) -> dict[str, int | str | float]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"b{self.batch}_n{self.heads}_s{self.seq_len}_d{self.head_dim}"

    @property
    def shape_summary(self) -> dict[str, object]:
        shape = [self.batch, self.heads, self.seq_len, self.head_dim]
        return {
            "query": shape,
            "key": shape,
            "value": shape,
            "scores": [self.seq_len, self.seq_len],
            "output": shape,
        }


VARIANT = DenseBnsdVariant()
VARIANTS = (
    DenseBnsdVariant(batch=1, heads=1, seq_len=32, head_dim=64, seed=0),
    DenseBnsdVariant(batch=1, heads=1, seq_len=64, head_dim=64, seed=1),
    DenseBnsdVariant(batch=1, heads=1, seq_len=32, head_dim=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _scale(variant: DenseBnsdVariant) -> float:
    return 1.0 / math.sqrt(variant.head_dim)


def _reference_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    variant: DenseBnsdVariant,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query.float(),
        key.float(),
        value.float(),
        dropout_p=0.0,
        is_causal=False,
        scale=_scale(variant),
    ).to(torch.float16).float()


def make_dense_bnsd_inputs(
    variant: DenseBnsdVariant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    shape = (variant.batch, variant.heads, variant.seq_len, variant.head_dim)
    query_cpu = torch.randn(shape, generator=generator, dtype=torch.float32).mul(variant.input_scale).to(torch.float16)
    key_cpu = torch.randn(shape, generator=generator, dtype=torch.float32).mul(variant.input_scale).to(torch.float16)
    value_cpu = torch.randn(shape, generator=generator, dtype=torch.float32).mul(variant.input_scale).to(torch.float16)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "query": query_cpu.npu(),
        "query_pto": query_cpu.mul(_scale(variant)).to(torch.float16).npu(),
        "key": key_cpu.npu(),
        "value": value_cpu.npu(),
        "key_t": key_cpu.transpose(-1, -2).contiguous().npu(),
        "scores_pto": torch.empty(
            (variant.seq_len, variant.seq_len), dtype=torch.float16
        ).npu(),
        "output_pto": torch.empty(shape, dtype=torch.float16).npu(),
        "reference": _reference_cpu(query_cpu, key_cpu, value_cpu, variant),
    }


def run_torch_npu_flash_attention_score(inputs: dict[str, object]):
    variant = inputs["variant"]
    return torch_npu.npu_fusion_attention_v2(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        variant["heads"],
        variant["input_layout"],
        scale=_scale(DenseBnsdVariant(**variant)),
        keep_prob=variant["keep_prob"],
        pre_tokens=2147483647,
        next_tokens=2147483647,
        sparse_mode=variant["sparse_mode"],
    )


def run_pto_flash_attention_score_variant(wrapper, inputs: dict[str, object]) -> torch.Tensor:
    wrapper(
        inputs["output_pto"],
        inputs["scores_pto"],
        inputs["query_pto"],
        inputs["key_t"],
        inputs["value"],
    )
    return inputs["output_pto"].float()
