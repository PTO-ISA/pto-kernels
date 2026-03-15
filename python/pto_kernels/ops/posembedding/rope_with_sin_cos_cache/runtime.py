"""Runtime helpers for the rope_with_sin_cos_cache migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


HEAD_DIM = 64
COS_SIN_DIM = HEAD_DIM * 2


@dataclass(frozen=True)
class RopeWithSinCosCacheVariant:
    batch: int = 2
    cache_len: int = 8
    seed: int = 0
    dtype: str = "float16"

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "positions": [self.batch],
            "query": [self.batch, HEAD_DIM],
            "key": [self.batch, HEAD_DIM],
            "cos_sin_cache": [self.cache_len, COS_SIN_DIM],
            "query_out": [self.batch, HEAD_DIM],
            "key_out": [self.batch, HEAD_DIM],
        }

    @property
    def label(self) -> str:
        return f"b{self.batch}_cache{self.cache_len}"


VARIANTS = (
    RopeWithSinCosCacheVariant(batch=2, cache_len=8, seed=0),
    RopeWithSinCosCacheVariant(batch=4, cache_len=8, seed=1),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _rotate_half_cpu(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_half_rope_cpu(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos + _rotate_half_cpu(x) * sin).float()


def _reference_cpu(
    query_cpu: torch.Tensor,
    key_cpu: torch.Tensor,
    positions_cpu: torch.Tensor,
    cos_sin_cache_cpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = cos_sin_cache_cpu.index_select(0, positions_cpu.to(torch.int64))
    cos_cpu = selected[:, :HEAD_DIM].float()
    sin_cpu = selected[:, HEAD_DIM:].float()
    query_ref = _apply_half_rope_cpu(query_cpu.float(), cos_cpu, sin_cpu)
    key_ref = _apply_half_rope_cpu(key_cpu.float(), cos_cpu, sin_cpu)
    return query_ref, key_ref


def make_inputs(variant: RopeWithSinCosCacheVariant, *, device_index: int = 0) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    query_cpu = torch.randn((variant.batch, HEAD_DIM), generator=generator, dtype=torch.float32).to(torch.float16)
    key_cpu = torch.randn((variant.batch, HEAD_DIM), generator=generator, dtype=torch.float32).to(torch.float16)
    cos_sin_cache_cpu = torch.randn((variant.cache_len, COS_SIN_DIM), generator=generator, dtype=torch.float32).to(torch.float16)
    positions_cpu = torch.arange(variant.batch, dtype=torch.int32)
    reference_query, reference_key = _reference_cpu(query_cpu, key_cpu, positions_cpu, cos_sin_cache_cpu)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "positions": positions_cpu.npu(),
        "query": query_cpu.npu(),
        "key": key_cpu.npu(),
        "cos_sin_cache": cos_sin_cache_cpu.npu(),
        "query_out": torch.empty_like(query_cpu).npu(),
        "key_out": torch.empty_like(key_cpu).npu(),
        "reference_query": reference_query,
        "reference_key": reference_key,
    }


def baseline_available() -> bool:
    return hasattr(torch_npu, "npu_rope_with_sin_cos_cache") or hasattr(torch.ops.npu, "npu_rope_with_sin_cos_cache")


def run_torch_npu_rope_with_sin_cos_cache(inputs: dict[str, object]):
    if hasattr(torch_npu, "npu_rope_with_sin_cos_cache"):
        return torch_npu.npu_rope_with_sin_cos_cache(
            inputs["positions"].to(torch.int64),
            inputs["query"],
            inputs["key"],
            inputs["cos_sin_cache"],
            [],
            HEAD_DIM,
            True,
        )
    return torch.ops.npu.npu_rope_with_sin_cos_cache(
        inputs["positions"].to(torch.int64),
        inputs["query"],
        inputs["key"],
        inputs["cos_sin_cache"],
        [],
        HEAD_DIM,
        True,
    )


def run_pto_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["query_out"],
        inputs["key_out"],
        inputs["query"],
        inputs["key"],
        inputs["cos_sin_cache"],
    )
    return inputs["query_out"], inputs["key_out"]
