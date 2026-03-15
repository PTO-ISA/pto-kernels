"""Runtime helpers for the first moe_token_permute_with_routing_map slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class RoutingMapPermutationVariant:
    tokens: int = 8
    hidden_size: int = 16
    experts: int = 4
    seed: int = 0
    dtype: str = "float16"
    routing_dtype: str = "int8"
    num_out_tokens: int = 8
    drop_and_pad: bool = False

    def as_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"t{self.tokens}_h{self.hidden_size}_e{self.experts}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "tokens": [self.tokens, self.hidden_size],
            "routing_map": [self.tokens, self.experts],
            "sorted_indices": [self.num_out_tokens],
            "gather_indices": [self.num_out_tokens * self.hidden_size],
        }


VARIANT = RoutingMapPermutationVariant()
VARIANTS = (
    RoutingMapPermutationVariant(tokens=8, hidden_size=16, experts=4, seed=0, num_out_tokens=8),
    RoutingMapPermutationVariant(tokens=256, hidden_size=64, experts=8, seed=1, num_out_tokens=256),
    RoutingMapPermutationVariant(tokens=128, hidden_size=128, experts=8, seed=2, num_out_tokens=128),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _build_top1_routing_map(variant: RoutingMapPermutationVariant, generator: torch.Generator) -> torch.Tensor:
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


def _routing_reference(tokens: torch.Tensor, routing_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_ids = torch.arange(tokens.shape[0], dtype=torch.int64)
    sorted_indices_first = token_ids.expand(routing_map.shape[1], -1).masked_select(routing_map.transpose(0, 1).to(torch.bool))
    sorted_indices = torch.argsort(sorted_indices_first).to(torch.int32)
    permuted_tokens = tokens.index_select(0, sorted_indices_first.to(torch.int64))
    gather_indices = (
        sorted_indices_first[:, None] * tokens.shape[1]
        + torch.arange(tokens.shape[1], dtype=torch.int64)[None, :]
    ).reshape(-1).to(torch.int32)
    return permuted_tokens, sorted_indices, gather_indices


def make_routing_map_permutation_inputs(
    variant: RoutingMapPermutationVariant = VARIANT,
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
    routing_map_cpu = _build_top1_routing_map(variant, generator)
    reference_tokens, reference_sorted_indices, gather_indices_cpu = _routing_reference(tokens_cpu.float(), routing_map_cpu)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "tokens": tokens_cpu.npu(),
        "routing_map": routing_map_cpu.npu(),
        "gather_indices": gather_indices_cpu.npu(),
        "sorted_indices_in": reference_sorted_indices.npu(),
        "permuted_tokens_pto": torch.empty((variant.num_out_tokens, variant.hidden_size), dtype=tokens_cpu.dtype).npu(),
        "sorted_indices_pto": torch.empty((variant.num_out_tokens,), dtype=torch.int32).npu(),
        "reference_tokens": reference_tokens.float(),
        "reference_sorted_indices": reference_sorted_indices.cpu(),
    }


def run_torch_npu_moe_token_permute_with_routing_map(inputs: dict[str, object]):
    return torch_npu.npu_moe_token_permute_with_routing_map(
        inputs["tokens"],
        inputs["routing_map"],
        probs=None,
        num_out_tokens=inputs["variant"]["num_out_tokens"],
        drop_and_pad=inputs["variant"]["drop_and_pad"],
    )


def run_pto_moe_token_permute_with_routing_map_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["permuted_tokens_pto"],
        inputs["sorted_indices_pto"],
        inputs["tokens"],
        inputs["gather_indices"],
        inputs["sorted_indices_in"],
    )
    return (
        inputs["permuted_tokens_pto"].float(),
        inputs["sorted_indices_pto"].to(torch.int32),
    )
