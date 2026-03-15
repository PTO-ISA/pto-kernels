"""Runtime helpers for the moe_token_permute_grad migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class TokenPermuteGradVariant:
    tokens: int
    hidden_size: int
    topk: int
    seed: int = 0
    dtype: str = "float16"
    padded_mode: bool = False
    probs: bool = False

    def as_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"t{self.tokens}_h{self.hidden_size}_k{self.topk}"

    @property
    def total_rows(self) -> int:
        return self.tokens * self.topk

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "tokens": [self.tokens, self.hidden_size],
            "grad_permuted_tokens": [self.total_rows, self.hidden_size],
            "indices": [self.tokens],
            "sorted_indices": [self.total_rows],
            "out": [self.tokens, self.hidden_size],
        }


VARIANT = TokenPermuteGradVariant(tokens=8, hidden_size=16, topk=1, seed=0)
VARIANTS = (
    TokenPermuteGradVariant(tokens=8, hidden_size=16, topk=1, seed=0),
    TokenPermuteGradVariant(tokens=256, hidden_size=64, topk=1, seed=1),
    TokenPermuteGradVariant(tokens=128, hidden_size=128, topk=1, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def make_token_permute_grad_inputs(
    variant: TokenPermuteGradVariant = VARIANT,
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
    grad_perm_cpu = torch.randn(
        (variant.total_rows, variant.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    indices_cpu = torch.randperm(variant.tokens, generator=generator, dtype=torch.int64).to(torch.int32)
    sorted_indices_cpu = torch.randperm(variant.total_rows, generator=generator, dtype=torch.int64).to(torch.int32)

    reference_out = grad_perm_cpu.index_select(0, sorted_indices_cpu.to(torch.int64)).float()

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "tokens": tokens_cpu.npu(),
        "grad_permuted_tokens": grad_perm_cpu.npu(),
        "indices": indices_cpu.npu(),
        "sorted_indices": sorted_indices_cpu.npu(),
        "out_pto": torch.empty_like(tokens_cpu).npu(),
        "reference_out": reference_out,
    }


def run_torch_npu_moe_token_permute_grad(inputs: dict[str, object]):
    return torch_npu.npu_moe_token_permute_grad(
        inputs["tokens"],
        inputs["grad_permuted_tokens"],
        inputs["indices"],
        inputs["sorted_indices"],
        inputs["variant"]["padded_mode"],
    )


def run_pto_moe_token_permute_grad_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["out_pto"],
        inputs["grad_permuted_tokens"],
        inputs["sorted_indices"],
    )
    return inputs["out_pto"].float()
