"""Runtime helpers for the first ring_attention_update migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class RingAttentionUpdateVariant:
    rows: int
    head_dim: int
    seed: int
    heads: int = 1
    sp: int = 2
    dtype: str = "float16"
    layout: str = "TND"

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"t{self.rows}_d{self.head_dim}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "prev_attn_out": [self.rows, self.heads, self.head_dim],
            "prev_softmax_max": [self.rows, self.heads, 8],
            "prev_softmax_sum": [self.rows, self.heads, 8],
            "cur_attn_out": [self.rows, self.heads, self.head_dim],
            "cur_softmax_max": [self.rows, self.heads, 8],
            "cur_softmax_sum": [self.rows, self.heads, 8],
            "attn_out": [self.rows, self.heads, self.head_dim],
            "softmax_max_out": [self.rows, self.heads, 8],
            "softmax_sum_out": [self.rows, self.heads, 8],
        }


VARIANT = RingAttentionUpdateVariant(rows=8, head_dim=16, seed=0)
VARIANTS = (
    RingAttentionUpdateVariant(rows=8, head_dim=16, seed=0),
    RingAttentionUpdateVariant(rows=256, head_dim=64, seed=1),
    RingAttentionUpdateVariant(rows=128, head_dim=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _reference_cpu(
    prev_attn_out: torch.Tensor,
    prev_softmax_max: torch.Tensor,
    prev_softmax_sum: torch.Tensor,
    cur_attn_out: torch.Tensor,
    cur_softmax_max: torch.Tensor,
    cur_softmax_sum: torch.Tensor,
) -> dict[str, torch.Tensor]:
    prev_attn_f32 = prev_attn_out.float()
    cur_attn_f32 = cur_attn_out.float()
    prev_max_f32 = prev_softmax_max[..., :1].float()
    prev_sum_f32 = prev_softmax_sum[..., :1].float()
    cur_max_f32 = cur_softmax_max[..., :1].float()
    cur_sum_f32 = cur_softmax_sum[..., :1].float()

    softmax_max = torch.maximum(prev_max_f32, cur_max_f32)
    prev_factor = torch.exp(prev_max_f32 - softmax_max) * prev_sum_f32
    cur_factor = torch.exp(cur_max_f32 - softmax_max) * cur_sum_f32
    softmax_sum = prev_factor + cur_factor

    prev_weight = prev_factor / softmax_sum
    cur_weight = cur_factor / softmax_sum
    attn_out = (prev_attn_f32 * prev_weight + cur_attn_f32 * cur_weight).to(torch.float16).float()
    softmax_max_out = softmax_max.expand(-1, -1, 8).contiguous().float()
    softmax_sum_out = softmax_sum.expand(-1, -1, 8).contiguous().float()
    return {
        "attn_out": attn_out,
        "softmax_max_out": softmax_max_out,
        "softmax_sum_out": softmax_sum_out,
    }


def make_ring_attention_update_inputs(
    variant: RingAttentionUpdateVariant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    prev_max_cpu = torch.randn((variant.rows, variant.heads, 1), generator=generator, dtype=torch.float32)
    cur_max_cpu = torch.randn((variant.rows, variant.heads, 1), generator=generator, dtype=torch.float32)
    prev_sum_cpu = torch.rand((variant.rows, variant.heads, 1), generator=generator, dtype=torch.float32) + 0.5
    cur_sum_cpu = torch.rand((variant.rows, variant.heads, 1), generator=generator, dtype=torch.float32) + 0.5
    prev_attn_cpu = torch.randn(
        (variant.rows, variant.heads, variant.head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    cur_attn_cpu = torch.randn(
        (variant.rows, variant.heads, variant.head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)

    prev_softmax_max_cpu = prev_max_cpu.expand(-1, -1, 8).contiguous()
    prev_softmax_sum_cpu = prev_sum_cpu.expand(-1, -1, 8).contiguous()
    cur_softmax_max_cpu = cur_max_cpu.expand(-1, -1, 8).contiguous()
    cur_softmax_sum_cpu = cur_sum_cpu.expand(-1, -1, 8).contiguous()

    reference = _reference_cpu(
        prev_attn_cpu,
        prev_softmax_max_cpu,
        prev_softmax_sum_cpu,
        cur_attn_cpu,
        cur_softmax_max_cpu,
        cur_softmax_sum_cpu,
    )

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "prev_attn_out": prev_attn_cpu.npu(),
        "prev_softmax_max": prev_softmax_max_cpu.npu(),
        "prev_softmax_sum": prev_softmax_sum_cpu.npu(),
        "cur_attn_out": cur_attn_cpu.npu(),
        "cur_softmax_max": cur_softmax_max_cpu.npu(),
        "cur_softmax_sum": cur_softmax_sum_cpu.npu(),
        "attn_out_pto": torch.empty((variant.rows, variant.heads, variant.head_dim), dtype=torch.float16).npu(),
        "softmax_max_out_pto": torch.empty((variant.rows, variant.heads, 8), dtype=torch.float32).npu(),
        "softmax_sum_out_pto": torch.empty((variant.rows, variant.heads, 8), dtype=torch.float32).npu(),
        "reference": reference,
    }


def baseline_available() -> bool:
    return bool(hasattr(torch, "ops") and hasattr(torch.ops, "npu") and hasattr(torch.ops.npu, "npu_ring_attention_update"))


def run_pto_ring_attention_update_variant(wrapper, inputs: dict[str, object]) -> dict[str, torch.Tensor]:
    wrapper(
        inputs["attn_out_pto"],
        inputs["softmax_max_out_pto"],
        inputs["softmax_sum_out_pto"],
        inputs["prev_attn_out"],
        inputs["prev_softmax_max"],
        inputs["prev_softmax_sum"],
        inputs["cur_attn_out"],
        inputs["cur_softmax_max"],
        inputs["cur_softmax_sum"],
    )
    return {
        "attn_out": inputs["attn_out_pto"].float(),
        "softmax_max_out": inputs["softmax_max_out_pto"].float(),
        "softmax_sum_out": inputs["softmax_sum_out_pto"].float(),
    }
