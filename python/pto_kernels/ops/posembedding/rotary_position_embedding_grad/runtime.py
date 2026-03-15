"""Runtime helpers for the rotary_position_embedding_grad migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


HEAD_DIM = 128
HALF_DIM = HEAD_DIM // 2


@dataclass(frozen=True)
class RotaryPositionEmbeddingGradVariant:
    layout: str
    batch: int = 2
    seq_len: int = 32
    heads: int = 1
    head_dim: int = HEAD_DIM
    seed: int = 0
    dtype: str = "float16"
    mode: str = "half"

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def shape_summary(self) -> dict[str, object]:
        shape = _shape_for(self)
        return {
            "dy": list(shape),
            "x": list(shape),
            "cos": list(shape),
            "sin": list(shape),
            "dx": list(shape),
            "dcos": list(shape),
            "dsin": list(shape),
        }

    @property
    def total_rows(self) -> int:
        return self.batch * self.seq_len * self.heads

    @property
    def label(self) -> str:
        return f"{self.layout.lower()}_b{self.batch}_s{self.seq_len}_n{self.heads}_d{self.head_dim}"


VARIANTS = (
    RotaryPositionEmbeddingGradVariant(layout="BSND", batch=2, seq_len=32, heads=1, seed=0),
    RotaryPositionEmbeddingGradVariant(layout="BNSD", batch=2, seq_len=32, heads=1, seed=1),
)


def _shape_for(variant: RotaryPositionEmbeddingGradVariant) -> tuple[int, ...]:
    if variant.layout == "BSND":
        return (variant.batch, variant.seq_len, variant.heads, variant.head_dim)
    if variant.layout == "BNSD":
        return (variant.batch, variant.heads, variant.seq_len, variant.head_dim)
    raise ValueError(f"Unsupported layout {variant.layout!r}")


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _rotate_half_cpu(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((-x[..., HALF_DIM:], x[..., :HALF_DIM]), dim=-1)


def _reference_cpu(
    dy_cpu: torch.Tensor,
    x_cpu: torch.Tensor,
    cos_cpu: torch.Tensor,
    sin_cpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dy1, dy2 = torch.chunk(dy_cpu.float(), 2, dim=-1)
    cos1, cos2 = torch.chunk(cos_cpu.float(), 2, dim=-1)
    sin1, sin2 = torch.chunk(sin_cpu.float(), 2, dim=-1)
    x1, x2 = torch.chunk(x_cpu.float(), 2, dim=-1)

    dx = torch.cat((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1)
    dcos = dy_cpu.float() * x_cpu.float()
    dsin = dy_cpu.float() * torch.cat((-x2, x1), dim=-1)
    return dx, dcos, dsin


def make_inputs(variant: RotaryPositionEmbeddingGradVariant, *, device_index: int = 0) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    shape = _shape_for(variant)
    dy_cpu = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.float16)
    x_cpu = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.float16)
    cos_cpu = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.float16)
    sin_cpu = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.float16)
    reference_dx, reference_dcos, reference_dsin = _reference_cpu(dy_cpu, x_cpu, cos_cpu, sin_cpu)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "dy": dy_cpu.npu(),
        "x": x_cpu.npu(),
        "cos": cos_cpu.npu(),
        "sin": sin_cpu.npu(),
        "reference_dx": reference_dx,
        "reference_dcos": reference_dcos,
        "reference_dsin": reference_dsin,
    }


def run_torch_npu_rotary_position_embedding_grad(inputs: dict[str, object]):
    return torch_npu.npu_rotary_mul_backward(
        inputs["dy"],
        inputs["x"],
        inputs["cos"],
        inputs["sin"],
        "half",
    )


def run_pto_variant(wrapper, inputs: dict[str, object]):
    return wrapper(
        inputs["dy"],
        inputs["x"],
        inputs["cos"],
        inputs["sin"],
        int(inputs["variant"]["head_dim"] and inputs["variant"]["batch"] * inputs["variant"]["seq_len"] * inputs["variant"]["heads"]),
    )
