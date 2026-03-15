"""Runtime helpers for the first moe_token_unpermute migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class Top1UnpermuteVariant:
    tokens: int
    hidden_size: int
    seed: int = 0
    dtype: str = "float16"
    padded_mode: bool = False
    index_dtype: str = "int32"
    probs: bool = False

    def as_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"t{self.tokens}_h{self.hidden_size}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "permuted_tokens": [self.tokens, self.hidden_size],
            "sorted_indices": [self.tokens],
            "gather_indices": [self.tokens * self.hidden_size],
            "out": [self.tokens, self.hidden_size],
        }


VARIANT = Top1UnpermuteVariant(tokens=8, hidden_size=16, seed=0)
VARIANTS = (
    VARIANT,
    Top1UnpermuteVariant(tokens=256, hidden_size=64, seed=1),
    Top1UnpermuteVariant(tokens=128, hidden_size=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _inverse_permutation(order: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(order, dtype=torch.int32)
    out[order] = torch.arange(order.numel(), dtype=torch.int32)
    return out


def make_top1_unpermute_inputs(
    variant: Top1UnpermuteVariant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    tokens_cpu = torch.randn(
        (variant.tokens, variant.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    perm_cpu = torch.randperm(variant.tokens, generator=generator, dtype=torch.int64)
    indices_cpu = perm_cpu.to(torch.int32)
    sorted_order = torch.argsort(indices_cpu.to(torch.int64))
    permuted_tokens_cpu = tokens_cpu.index_select(0, sorted_order)
    sorted_indices_cpu = _inverse_permutation(sorted_order)
    gather_indices_cpu = (
        sorted_indices_cpu.to(torch.int64)[:, None] * variant.hidden_size
        + torch.arange(variant.hidden_size, dtype=torch.int64)[None, :]
    ).reshape(-1).to(torch.int32)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "permuted_tokens": permuted_tokens_cpu.npu(),
        "sorted_indices": sorted_indices_cpu.npu(),
        "gather_indices": gather_indices_cpu.npu(),
        "restored_tokens_pto": torch.empty_like(tokens_cpu).npu(),
        "reference_tokens": tokens_cpu.float(),
    }


def run_torch_npu_moe_token_unpermute(inputs: dict[str, object]):
    return torch_npu.npu_moe_token_unpermute(
        inputs["permuted_tokens"],
        inputs["sorted_indices"],
        None,
        padded_mode=inputs["variant"]["padded_mode"],
        restore_shape=None,
    )


def run_pto_moe_token_unpermute_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["restored_tokens_pto"],
        inputs["permuted_tokens"],
        inputs["gather_indices"],
    )
    return inputs["restored_tokens_pto"].float()
