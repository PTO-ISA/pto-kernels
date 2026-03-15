"""Runtime helpers for the constrained SwinAttentionFFN PTO slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class SwinAttentionFfnVariant:
    batch: int
    seq: int
    hidden: int
    seed: int = 0
    input_scale: float = 0.125
    dtype: str = "float16"
    shift1: int = 0
    shift2: int = 0

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)

    @property
    def tokens(self) -> int:
        return self.batch * self.seq

    @property
    def label(self) -> str:
        return f"b{self.batch}_s{self.seq}_h{self.hidden}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x1": [self.batch, self.seq, self.hidden],
            "x2": [self.hidden, self.hidden],
            "bias": [self.hidden],
            "x3": [self.batch, self.seq, self.hidden],
            "y": [self.batch, self.seq, self.hidden],
            "flattened_tokens": self.tokens,
        }


VARIANTS = (
    SwinAttentionFfnVariant(batch=2, seq=64, hidden=128, seed=0),
    SwinAttentionFfnVariant(batch=48, seq=64, hidden=128, seed=1),
    SwinAttentionFfnVariant(batch=8, seq=64, hidden=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def make_inputs(
    variant: SwinAttentionFfnVariant,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    x1_cpu = (
        torch.randn((variant.batch, variant.seq, variant.hidden), generator=generator, dtype=torch.float32)
        .mul(variant.input_scale)
        .to(torch.float16)
    )
    x2_cpu = (
        torch.randn((variant.hidden, variant.hidden), generator=generator, dtype=torch.float32)
        .mul(variant.input_scale)
        .to(torch.float16)
    )
    bias_cpu = (
        torch.randn((variant.hidden,), generator=generator, dtype=torch.float32)
        .mul(variant.input_scale)
        .to(torch.float16)
    )
    x3_cpu = (
        torch.randn((variant.batch, variant.seq, variant.hidden), generator=generator, dtype=torch.float32)
        .mul(variant.input_scale)
        .to(torch.float16)
    )

    x1_flat = x1_cpu.reshape(variant.tokens, variant.hidden)
    x3_flat = x3_cpu.reshape(variant.tokens, variant.hidden)
    reference = (x1_flat.float() @ x2_cpu.float() + bias_cpu.float() + x3_flat.float()).cpu()

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "x1": x1_flat.npu(),
        "x2": x2_cpu.npu(),
        "bias": bias_cpu.npu(),
        "x3": x3_flat.npu(),
        "tmp_pto": torch.empty((variant.tokens, variant.hidden), dtype=torch.float16).npu(),
        "out_pto": torch.empty((variant.tokens, variant.hidden), dtype=torch.float16).npu(),
        "reference": reference,
    }


def baseline_available() -> bool:
    try:
        import torch_npu
    except Exception:
        return False
    return hasattr(torch_npu, "npu_swin_attention_ffn") or hasattr(torch.ops.npu, "npu_swin_attention_ffn")


def run_pto_variant(wrapper, inputs: dict[str, object]) -> torch.Tensor:
    wrapper(
        inputs["out_pto"],
        inputs["tmp_pto"],
        inputs["x1"],
        inputs["x2"],
        inputs["bias"],
        inputs["x3"],
    )
    return inputs["out_pto"].float()
