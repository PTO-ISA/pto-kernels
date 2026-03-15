"""Runtime helpers for the first attention_update migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class AttentionUpdateVariant:
    rows: int
    head_dim: int
    seed: int
    sp: int = 2
    dtype: str = "float16"
    update_type: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"r{self.rows}_d{self.head_dim}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "lse0": [self.rows],
            "lse1": [self.rows],
            "local_out0": [self.rows, self.head_dim],
            "local_out1": [self.rows, self.head_dim],
            "out": [self.rows, self.head_dim],
        }


VARIANT = AttentionUpdateVariant(rows=8, head_dim=16, seed=0)
VARIANTS = (
    AttentionUpdateVariant(rows=8, head_dim=16, seed=0),
    AttentionUpdateVariant(rows=256, head_dim=64, seed=1),
    AttentionUpdateVariant(rows=128, head_dim=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _reference_cpu(lse0: torch.Tensor, lse1: torch.Tensor, local_out0: torch.Tensor, local_out1: torch.Tensor) -> torch.Tensor:
    lse0_f32 = lse0.float()
    lse1_f32 = lse1.float()
    local_out0_f32 = local_out0.float()
    local_out1_f32 = local_out1.float()

    lse_max = torch.maximum(lse0_f32, lse1_f32)
    w0 = torch.exp(lse0_f32 - lse_max)
    w1 = torch.exp(lse1_f32 - lse_max)
    denom = w0 + w1
    w0 = (w0 / denom).unsqueeze(-1)
    w1 = (w1 / denom).unsqueeze(-1)
    return (local_out0_f32 * w0 + local_out1_f32 * w1).to(torch.float16).float()


def make_attention_update_inputs(variant: AttentionUpdateVariant = VARIANT, *, device_index: int = 0) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    lse0_cpu = torch.randn((variant.rows,), generator=generator, dtype=torch.float32)
    lse1_cpu = torch.randn((variant.rows,), generator=generator, dtype=torch.float32)
    local_out0_cpu = torch.randn((variant.rows, variant.head_dim), generator=generator, dtype=torch.float32).to(torch.float16)
    local_out1_cpu = torch.randn((variant.rows, variant.head_dim), generator=generator, dtype=torch.float32).to(torch.float16)
    reference = _reference_cpu(lse0_cpu, lse1_cpu, local_out0_cpu, local_out1_cpu)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "lse0": lse0_cpu.npu(),
        "lse1": lse1_cpu.npu(),
        "local_out0": local_out0_cpu.npu(),
        "local_out1": local_out1_cpu.npu(),
        "out_pto": torch.empty((variant.rows, variant.head_dim), dtype=torch.float16).npu(),
        "reference": reference,
    }


def run_torch_npu_attention_update(inputs: dict[str, object]):
    return torch_npu.npu_attention_update(
        [inputs["lse0"], inputs["lse1"]],
        [inputs["local_out0"], inputs["local_out1"]],
        int(inputs["variant"]["update_type"]),
    )


def run_pto_attention_update_variant(wrapper, inputs: dict[str, object]) -> torch.Tensor:
    wrapper(
        inputs["out_pto"],
        inputs["lse0"],
        inputs["lse1"],
        inputs["local_out0"],
        inputs["local_out1"],
    )
    return inputs["out_pto"].float()
