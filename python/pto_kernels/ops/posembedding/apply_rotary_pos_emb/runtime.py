"""Runtime helpers for constrained apply_rotary_pos_emb migration slices."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class RopeVariant:
    layout: str
    head_dim: int = 128
    seed: int = 0
    dtype: str = "float16"
    rotary_mode: str = "half"
    tokens: int = 64
    batch: int = 1
    seq_len: int = 64
    heads: int = 1

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def shape_summary(self) -> dict[str, object]:
        query_shape, rope_shape = _shape_for(self)
        return {
            "query": list(query_shape),
            "key": list(query_shape),
            "cos": list(rope_shape),
            "sin": list(rope_shape),
        }

    @property
    def total_rows(self) -> int:
        if self.layout == "TND":
            return self.tokens * self.heads
        if self.layout == "BSND":
            return self.batch * self.seq_len * self.heads
        raise ValueError(f"Unsupported layout {self.layout!r}")

TND_HALF_VARIANT = RopeVariant(layout="TND", tokens=64, batch=1, seq_len=64, heads=1)
BSND_HALF_VARIANT = RopeVariant(layout="BSND", tokens=64, batch=2, seq_len=32, heads=1)
VARIANT = TND_HALF_VARIANT
VARIANTS = (TND_HALF_VARIANT, BSND_HALF_VARIANT)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _rotate_half_cpu(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _reference_cpu(
    query_cpu: torch.Tensor,
    key_cpu: torch.Tensor,
    cos_cpu: torch.Tensor,
    sin_cpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_ref = (query_cpu * cos_cpu + _rotate_half_cpu(query_cpu) * sin_cpu).float()
    k_ref = (key_cpu * cos_cpu + _rotate_half_cpu(key_cpu) * sin_cpu).float()
    return q_ref, k_ref


def _shape_for(variant: RopeVariant) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if variant.layout == "TND":
        return (
            (variant.tokens, variant.heads, variant.head_dim),
            (variant.tokens, 1, variant.head_dim),
        )
    if variant.layout == "BSND":
        return (
            (variant.batch, variant.seq_len, variant.heads, variant.head_dim),
            (variant.batch, variant.seq_len, 1, variant.head_dim),
        )
    raise ValueError(f"Unsupported layout {variant.layout!r}")


def make_inputs(variant: RopeVariant, *, device_index: int = 0) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    query_shape, rope_shape = _shape_for(variant)
    query_cpu = torch.randn(query_shape, generator=generator, dtype=torch.float32).to(torch.float16)
    key_cpu = torch.randn(query_shape, generator=generator, dtype=torch.float32).to(torch.float16)
    cos_cpu = torch.randn(rope_shape, generator=generator, dtype=torch.float32).to(torch.float16)
    sin_cpu = torch.randn(rope_shape, generator=generator, dtype=torch.float32).to(torch.float16)

    query_ref, key_ref = _reference_cpu(query_cpu, key_cpu, cos_cpu, sin_cpu)

    query_src = query_cpu.npu()
    key_src = key_cpu.npu()
    cos = cos_cpu.npu()
    sin = sin_cpu.npu()
    query = query_src.clone()
    key = key_src.clone()

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "query_src": query_src,
        "key_src": key_src,
        "query": query,
        "key": key,
        "cos": cos,
        "sin": sin,
        "rows": variant.total_rows,
        "reference_query": query_ref,
        "reference_key": key_ref,
    }


def make_tnd_half_inputs(*, device_index: int = 0) -> dict[str, object]:
    return make_inputs(TND_HALF_VARIANT, device_index=device_index)


def make_bsnd_half_inputs(*, device_index: int = 0) -> dict[str, object]:
    return make_inputs(BSND_HALF_VARIANT, device_index=device_index)


def reset_inplace_inputs(inputs: dict[str, object]) -> None:
    inputs["query"].copy_(inputs["query_src"])
    inputs["key"].copy_(inputs["key_src"])


def run_torch_npu_apply_rotary_pos_emb(inputs: dict[str, object]):
    variant = inputs["variant"]
    return torch_npu.npu_apply_rotary_pos_emb(
        inputs["query"],
        inputs["key"],
        inputs["cos"],
        inputs["sin"],
        layout=variant["layout"],
        rotary_mode=variant["rotary_mode"],
    )


def run_pto_variant(wrapper, inputs: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    wrapper(
        inputs["query"],
        inputs["key"],
        inputs["cos"],
        inputs["sin"],
        int(inputs["rows"]),
    )
    return inputs["query"], inputs["key"]


def run_pto_tnd_half_variant(wrapper, inputs: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    return run_pto_variant(wrapper, inputs)
