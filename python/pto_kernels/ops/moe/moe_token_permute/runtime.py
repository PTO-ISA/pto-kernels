"""Runtime helpers for the first moe_token_permute migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class Top1PermutationVariant:
    tokens: int = 8
    hidden_size: int = 16
    seed: int = 0
    dtype: str = "float16"
    num_out_tokens: int = 0
    padded_mode: bool = False
    index_dtype: str = "int32"

    def as_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)


VARIANT = Top1PermutationVariant()


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _inverse_permutation(order: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(order, dtype=torch.int32)
    out[order] = torch.arange(order.numel(), dtype=torch.int32)
    return out


def make_top1_permutation_inputs(*, device_index: int = 0) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(VARIANT.seed)

    tokens_cpu = torch.randn(
        (VARIANT.tokens, VARIANT.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    perm_cpu = torch.randperm(VARIANT.tokens, generator=generator, dtype=torch.int64)
    indices_cpu = perm_cpu.to(torch.int32)
    sorted_order = torch.argsort(indices_cpu.to(torch.int64))
    gather_indices_cpu = (
        sorted_order[:, None] * VARIANT.hidden_size
        + torch.arange(VARIANT.hidden_size, dtype=torch.int64)[None, :]
    ).reshape(-1).to(torch.int32)

    return {
        "device": device,
        "tokens": tokens_cpu.npu(),
        "indices": indices_cpu.npu(),
        "gather_indices": gather_indices_cpu.npu(),
        "permuted_tokens_pto": torch.empty_like(tokens_cpu).npu(),
        "sorted_indices_pto": torch.empty_like(indices_cpu).npu(),
        "reference_tokens": tokens_cpu.index_select(0, sorted_order).float(),
        "reference_sorted_indices": _inverse_permutation(sorted_order).cpu(),
    }


def run_torch_npu_moe_token_permute(inputs: dict[str, object]):
    return torch_npu.npu_moe_token_permute(
        inputs["tokens"],
        inputs["indices"],
        num_out_tokens=VARIANT.num_out_tokens,
        padded_mode=VARIANT.padded_mode,
    )


def run_pto_moe_token_permute_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["permuted_tokens_pto"],
        inputs["sorted_indices_pto"],
        inputs["tokens"],
        inputs["indices"],
        inputs["gather_indices"],
    )
    return (
        inputs["permuted_tokens_pto"].float(),
        inputs["sorted_indices_pto"].to(torch.int32),
    )
