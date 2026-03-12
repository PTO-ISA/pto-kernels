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


VARIANT = DenseBnsdVariant()


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _scale() -> float:
    return 1.0 / math.sqrt(VARIANT.head_dim)


def _reference_cpu(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query.float(),
        key.float(),
        value.float(),
        dropout_p=0.0,
        is_causal=False,
        scale=_scale(),
    ).to(torch.float16).float()


def make_dense_bnsd_inputs(*, device_index: int = 0) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(VARIANT.seed)

    shape = (VARIANT.batch, VARIANT.heads, VARIANT.seq_len, VARIANT.head_dim)
    query_cpu = torch.randn(shape, generator=generator, dtype=torch.float32).mul(VARIANT.input_scale).to(torch.float16)
    key_cpu = torch.randn(shape, generator=generator, dtype=torch.float32).mul(VARIANT.input_scale).to(torch.float16)
    value_cpu = torch.randn(shape, generator=generator, dtype=torch.float32).mul(VARIANT.input_scale).to(torch.float16)

    return {
        "device": device,
        "query": query_cpu.npu(),
        "query_pto": query_cpu.mul(_scale()).to(torch.float16).npu(),
        "key": key_cpu.npu(),
        "value": value_cpu.npu(),
        "key_t": key_cpu.transpose(-1, -2).contiguous().npu(),
        "scores_pto": torch.empty(
            (VARIANT.seq_len, VARIANT.seq_len), dtype=torch.float16
        ).npu(),
        "output_pto": torch.empty(shape, dtype=torch.float16).npu(),
        "reference": _reference_cpu(query_cpu, key_cpu, value_cpu),
    }


def run_torch_npu_flash_attention_score(inputs: dict[str, object]):
    return torch_npu.npu_fusion_attention_v2(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        VARIANT.heads,
        VARIANT.input_layout,
        scale=_scale(),
        keep_prob=VARIANT.keep_prob,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        sparse_mode=VARIANT.sparse_mode,
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
