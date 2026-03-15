"""Runtime helpers for the rotary_position_embedding migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


HEAD_DIM = 128


@dataclass(frozen=True)
class RotaryPositionEmbeddingVariant:
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
            "x": list(shape),
            "cos": list(shape),
            "sin": list(shape),
            "out": list(shape),
        }

    @property
    def total_rows(self) -> int:
        return self.batch * self.seq_len * self.heads

    @property
    def label(self) -> str:
        return f"{self.layout.lower()}_b{self.batch}_s{self.seq_len}_n{self.heads}_d{self.head_dim}"


VARIANTS = (
    RotaryPositionEmbeddingVariant(layout="BSND", batch=2, seq_len=32, heads=1, seed=0),
    RotaryPositionEmbeddingVariant(layout="BNSD", batch=2, seq_len=32, heads=1, seed=1),
)


def _shape_for(variant: RotaryPositionEmbeddingVariant) -> tuple[int, ...]:
    if variant.layout == "BSND":
        return (variant.batch, variant.seq_len, variant.heads, variant.head_dim)
    if variant.layout == "BNSD":
        return (variant.batch, variant.heads, variant.seq_len, variant.head_dim)
    raise ValueError(f"Unsupported layout {variant.layout!r}")


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _rotate_half_cpu(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _reference_cpu(x_cpu: torch.Tensor, cos_cpu: torch.Tensor, sin_cpu: torch.Tensor) -> torch.Tensor:
    return (x_cpu.float() * cos_cpu.float() + _rotate_half_cpu(x_cpu.float()) * sin_cpu.float()).float()


def make_inputs(variant: RotaryPositionEmbeddingVariant, *, device_index: int = 0) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    shape = _shape_for(variant)
    x_cpu = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.float16)
    cos_cpu = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.float16)
    sin_cpu = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.float16)
    reference = _reference_cpu(x_cpu, cos_cpu, sin_cpu)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "x": x_cpu.npu(),
        "cos": cos_cpu.npu(),
        "sin": sin_cpu.npu(),
        "out": torch.empty_like(x_cpu).npu(),
        "rows": variant.total_rows,
        "reference": reference,
    }


def run_torch_npu_rotary_position_embedding(inputs: dict[str, object]) -> torch.Tensor:
    return torch_npu.npu_rotary_mul(inputs["x"], inputs["cos"], inputs["sin"], "half")


def run_pto_variant(wrapper, inputs: dict[str, object]) -> torch.Tensor:
    wrapper(
        inputs["out"],
        inputs["x"],
        inputs["cos"],
        inputs["sin"],
        int(inputs["rows"]),
    )
    return inputs["out"]
