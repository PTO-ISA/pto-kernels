"""Runtime helpers for the first moe_token_permute_with_routing_map_grad slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class RoutingMapPermutationGradVariant:
    tokens: int = 8
    hidden_size: int = 16
    experts: int = 4
    seed: int = 0
    dtype: str = "float16"
    routing_dtype: str = "int8"
    drop_and_pad: bool = False
    probs: bool = False

    def as_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"t{self.tokens}_h{self.hidden_size}_e{self.experts}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "permuted_tokens_grad": [self.tokens, self.hidden_size],
            "routing_map": [self.tokens, self.experts],
            "sorted_indices": [self.tokens],
            "token_grad_out": [self.tokens, self.hidden_size],
        }


VARIANT = RoutingMapPermutationGradVariant()
VARIANTS = (
    RoutingMapPermutationGradVariant(tokens=8, hidden_size=16, experts=4, seed=0),
    RoutingMapPermutationGradVariant(tokens=256, hidden_size=64, experts=8, seed=1),
    RoutingMapPermutationGradVariant(tokens=128, hidden_size=128, experts=8, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _build_top1_routing_map(variant: RoutingMapPermutationGradVariant, generator: torch.Generator) -> torch.Tensor:
    assignments = torch.randint(
        low=0,
        high=variant.experts,
        size=(variant.tokens,),
        generator=generator,
        dtype=torch.int64,
    )
    routing = torch.zeros((variant.tokens, variant.experts), dtype=torch.int8)
    routing[torch.arange(variant.tokens), assignments] = 1
    return routing


def _sorted_indices_from_routing_map(routing_map: torch.Tensor) -> torch.Tensor:
    token_ids = torch.arange(routing_map.shape[0], dtype=torch.int64)
    sorted_indices_first = token_ids.expand(routing_map.shape[1], -1).masked_select(
        routing_map.transpose(0, 1).to(torch.bool)
    )
    return torch.argsort(sorted_indices_first).to(torch.int32)


def make_routing_map_permutation_grad_inputs(
    variant: RoutingMapPermutationGradVariant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    permuted_grad_cpu = torch.randn(
        (variant.tokens, variant.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    routing_map_cpu = _build_top1_routing_map(variant, generator)
    sorted_indices_cpu = _sorted_indices_from_routing_map(routing_map_cpu)
    reference_token_grad_cpu = torch.zeros_like(permuted_grad_cpu, dtype=torch.float32)
    reference_token_grad_cpu[sorted_indices_cpu.to(torch.int64)] = permuted_grad_cpu.float()

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "permuted_tokens_grad": permuted_grad_cpu.npu(),
        "routing_map": routing_map_cpu.npu(),
        "sorted_indices": sorted_indices_cpu.npu(),
        "token_grad_pto": torch.empty_like(permuted_grad_cpu).npu(),
        "probs_grad_pto": torch.zeros((), dtype=torch.float16).npu(),
        "reference_token_grad": reference_token_grad_cpu,
        "reference_probs_grad": torch.tensor(0.0, dtype=torch.float32),
    }


def run_torch_npu_moe_token_permute_with_routing_map_grad(inputs: dict[str, object]):
    return torch_npu.npu_moe_token_permute_with_routing_map_grad(
        inputs["permuted_tokens_grad"],
        None,
        inputs["sorted_indices"],
        inputs["routing_map"],
        int(inputs["variant"]["experts"]),
        int(inputs["variant"]["tokens"]),
        bool(inputs["variant"]["drop_and_pad"]),
    )


def run_pto_moe_token_permute_with_routing_map_grad_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["token_grad_pto"],
        inputs["probs_grad_pto"],
        inputs["permuted_tokens_grad"],
        inputs["sorted_indices"],
    )
    return inputs["token_grad_pto"].float(), inputs["probs_grad_pto"].float()
