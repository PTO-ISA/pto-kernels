"""Runtime helpers for the interleave_rope migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class InterleaveRopeVariant:
    batch: int = 1
    heads: int = 1
    seq_len: int = 32
    head_dim: int = 64
    seed: int = 0
    dtype: str = "float16"

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def shape_summary(self) -> dict[str, object]:
        x_shape = [self.batch, self.heads, self.seq_len, self.head_dim]
        rope_shape = [self.batch, 1, self.seq_len, self.head_dim]
        return {
            "x": x_shape,
            "cos": rope_shape,
            "sin": rope_shape,
            "y": x_shape,
        }

    @property
    def total_rows(self) -> int:
        return self.batch * self.heads * self.seq_len

    @property
    def label(self) -> str:
        return f"b{self.batch}_n{self.heads}_s{self.seq_len}_d{self.head_dim}"


VARIANT = InterleaveRopeVariant()
VARIANTS = (
    InterleaveRopeVariant(batch=1, heads=1, seq_len=32, head_dim=64, seed=0),
    InterleaveRopeVariant(batch=2, heads=1, seq_len=32, head_dim=64, seed=1),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _rotate_half_cpu(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _reference_cpu(x_cpu: torch.Tensor, cos_cpu: torch.Tensor, sin_cpu: torch.Tensor) -> torch.Tensor:
    interleaved = x_cpu.reshape(*x_cpu.shape[:-1], x_cpu.shape[-1] // 2, 2).transpose(-1, -2).reshape_as(x_cpu)
    return (interleaved * cos_cpu + _rotate_half_cpu(interleaved) * sin_cpu).float()


def interleave_npu(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2).transpose(-1, -2).contiguous().reshape_as(x)


def make_inputs(variant: InterleaveRopeVariant, *, device_index: int = 0) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    x_shape = (variant.batch, variant.heads, variant.seq_len, variant.head_dim)
    rope_shape = (variant.batch, 1, variant.seq_len, variant.head_dim)
    x_cpu = torch.randn(x_shape, generator=generator, dtype=torch.float32).to(torch.float16)
    cos_cpu = torch.randn(rope_shape, generator=generator, dtype=torch.float32).to(torch.float16)
    sin_cpu = torch.randn(rope_shape, generator=generator, dtype=torch.float32).to(torch.float16)
    reference = _reference_cpu(x_cpu, cos_cpu, sin_cpu)

    x = x_cpu.npu()
    cos = cos_cpu.npu()
    sin = sin_cpu.npu()
    out = torch.empty_like(x)
    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "x": x,
        "cos": cos,
        "sin": sin,
        "out": out,
        "rows": variant.total_rows,
        "reference": reference,
    }


def run_torch_npu_interleave_rope(inputs: dict[str, object]) -> torch.Tensor:
    return torch_npu.npu_interleave_rope(inputs["x"], inputs["cos"], inputs["sin"])


def run_pto_variant(wrapper, inputs: dict[str, object]) -> torch.Tensor:
    wrapper(
        inputs["out"],
        inputs["x"],
        inputs["cos"],
        inputs["sin"],
        int(inputs["rows"]),
    )
    return inputs["out"]
