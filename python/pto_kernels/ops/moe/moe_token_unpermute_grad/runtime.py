"""Runtime helpers for the first moe_token_unpermute_grad migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class Top1UnpermuteGradVariant:
    tokens: int
    hidden_size: int
    seed: int = 0
    dtype: str = "float16"
    padded_mode: bool = False
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
            "unpermuted_tokens_grad": [self.tokens, self.hidden_size],
            "sorted_indices": [self.tokens],
            "permuted_tokens_grad": [self.tokens, self.hidden_size],
            "probs_grad": [],
        }


VARIANT = Top1UnpermuteGradVariant(tokens=8, hidden_size=16, seed=0)
VARIANTS = (
    VARIANT,
    Top1UnpermuteGradVariant(tokens=256, hidden_size=64, seed=1),
    Top1UnpermuteGradVariant(tokens=128, hidden_size=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def make_top1_unpermute_grad_inputs(
    variant: Top1UnpermuteGradVariant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    permuted_tokens_cpu = torch.randn(
        (variant.tokens, variant.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    unpermuted_grad_cpu = torch.randn(
        (variant.tokens, variant.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    sorted_indices_cpu = torch.randperm(variant.tokens, generator=generator, dtype=torch.int64).to(torch.int32)

    permuted_grad_ref = torch.empty_like(unpermuted_grad_cpu, dtype=torch.float32)
    permuted_grad_ref[sorted_indices_cpu.to(torch.int64)] = unpermuted_grad_cpu.float()
    probs_grad_ref = torch.tensor(0.0, dtype=torch.float32)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "permuted_tokens": permuted_tokens_cpu.npu(),
        "unpermuted_tokens_grad": unpermuted_grad_cpu.npu(),
        "sorted_indices": sorted_indices_cpu.npu(),
        "permuted_tokens_grad_pto": torch.empty_like(unpermuted_grad_cpu).npu(),
        "probs_grad_pto": torch.zeros((), dtype=torch.float16).npu(),
        "reference_permuted_tokens_grad": permuted_grad_ref,
        "reference_probs_grad": probs_grad_ref,
    }


def run_torch_npu_moe_token_unpermute_grad(inputs: dict[str, object]):
    return torch_npu.npu_moe_token_unpermute_grad(
        inputs["permuted_tokens"],
        inputs["unpermuted_tokens_grad"],
        inputs["sorted_indices"],
        None,
        inputs["variant"]["padded_mode"],
        None,
    )


def run_pto_moe_token_unpermute_grad_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["permuted_tokens_grad_pto"],
        inputs["probs_grad_pto"],
        inputs["unpermuted_tokens_grad"],
        inputs["sorted_indices"],
    )
    return inputs["permuted_tokens_grad_pto"].float(), inputs["probs_grad_pto"].float()
