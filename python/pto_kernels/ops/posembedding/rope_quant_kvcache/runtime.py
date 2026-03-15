"""Runtime helpers for the rope_quant_kvcache migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


SIZE_SPLITS = (64, 64, 64)
HEAD_DIM = 64


@dataclass(frozen=True)
class RopeQuantKvcacheVariant:
    batch: int = 2
    cache_len: int = 8
    seed: int = 0
    dtype: str = "float16"

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x": [self.batch, sum(SIZE_SPLITS)],
            "cos": [self.batch, HEAD_DIM],
            "sin": [self.batch, HEAD_DIM],
            "k_cache": [self.batch, self.cache_len, 1, HEAD_DIM],
            "v_cache": [self.batch, self.cache_len, 1, HEAD_DIM],
            "q": [self.batch, 1, HEAD_DIM],
            "k": [self.batch, 1, HEAD_DIM],
            "v": [self.batch, 1, HEAD_DIM],
        }

    @property
    def label(self) -> str:
        return f"b{self.batch}_cache{self.cache_len}"


VARIANTS = (
    RopeQuantKvcacheVariant(batch=2, cache_len=8, seed=0),
    RopeQuantKvcacheVariant(batch=4, cache_len=8, seed=1),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _rotate_half_cpu(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_half_rope_cpu(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos + _rotate_half_cpu(x) * sin).float()


def _quantize_unit_scale_cpu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(x), -128, 127).to(torch.int8)


def _reference_cpu(
    x_cpu: torch.Tensor,
    cos_cpu: torch.Tensor,
    sin_cpu: torch.Tensor,
    indices_cpu: torch.Tensor,
    cache_len: int,
) -> dict[str, torch.Tensor]:
    q_raw, k_raw, v_raw = torch.split(x_cpu.float(), SIZE_SPLITS, dim=-1)
    q_out = _apply_half_rope_cpu(q_raw, cos_cpu.float(), sin_cpu.float()).view(x_cpu.shape[0], 1, HEAD_DIM)
    k_out = _apply_half_rope_cpu(k_raw, cos_cpu.float(), sin_cpu.float()).view(x_cpu.shape[0], 1, HEAD_DIM)
    v_out = v_raw.float().view(x_cpu.shape[0], 1, HEAD_DIM)

    k_cache = torch.zeros((x_cpu.shape[0], cache_len, 1, HEAD_DIM), dtype=torch.int8)
    v_cache = torch.zeros((x_cpu.shape[0], cache_len, 1, HEAD_DIM), dtype=torch.int8)
    for batch_idx, cache_idx in enumerate(indices_cpu.tolist()):
        k_cache[batch_idx, cache_idx, 0, :] = _quantize_unit_scale_cpu(k_out[batch_idx, 0, :])
        v_cache[batch_idx, cache_idx, 0, :] = _quantize_unit_scale_cpu(v_out[batch_idx, 0, :])

    return {
        "q": q_out,
        "k": k_out,
        "v": v_out,
        "k_cache": k_cache,
        "v_cache": v_cache,
    }


def make_inputs(variant: RopeQuantKvcacheVariant, *, device_index: int = 0) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    x_cpu = torch.randn((variant.batch, sum(SIZE_SPLITS)), generator=generator, dtype=torch.float32).to(torch.float16)
    cos_cpu = torch.randn((variant.batch, HEAD_DIM), generator=generator, dtype=torch.float32).to(torch.float16)
    sin_cpu = torch.randn((variant.batch, HEAD_DIM), generator=generator, dtype=torch.float32).to(torch.float16)
    indices_cpu = torch.arange(variant.batch, dtype=torch.int32) % variant.cache_len
    reference = _reference_cpu(x_cpu, cos_cpu, sin_cpu, indices_cpu, variant.cache_len)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "x": x_cpu.npu(),
        "cos": cos_cpu.npu(),
        "sin": sin_cpu.npu(),
        "indices": indices_cpu.npu(),
        "scale_k": torch.ones((HEAD_DIM,), dtype=torch.float32).npu(),
        "scale_v": torch.ones((HEAD_DIM,), dtype=torch.float32).npu(),
        "k_cache": torch.zeros((variant.batch, variant.cache_len, 1, HEAD_DIM), dtype=torch.int8).npu(),
        "v_cache": torch.zeros((variant.batch, variant.cache_len, 1, HEAD_DIM), dtype=torch.int8).npu(),
        "reference": reference,
    }


def run_torch_npu_rope_quant_kvcache(inputs: dict[str, object]):
    return torch_npu.npu_rope_quant_kvcache(
        inputs["x"],
        inputs["cos"],
        inputs["sin"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["indices"],
        inputs["scale_k"],
        inputs["scale_v"],
        SIZE_SPLITS,
        kv_output=True,
        input_layout="BSND",
        cache_mode="contiguous",
    )


def run_pto_variant(wrapper, inputs: dict[str, object]):
    return wrapper(
        inputs["x"],
        inputs["cos"],
        inputs["sin"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["indices"],
        inputs["scale_k"],
        inputs["scale_v"],
    )
