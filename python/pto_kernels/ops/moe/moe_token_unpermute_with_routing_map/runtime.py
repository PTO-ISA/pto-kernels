"""Runtime helpers for the first moe_token_unpermute_with_routing_map slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class RoutingMapUnpermuteVariant:
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
            "permuted_tokens": [self.tokens, self.hidden_size],
            "routing_map": [self.tokens, self.experts],
            "sorted_indices": [self.tokens],
            "gather_indices": [self.tokens * self.hidden_size],
            "restore_shape": [self.tokens, self.hidden_size],
        }


VARIANT = RoutingMapUnpermuteVariant()
VARIANTS = (
    RoutingMapUnpermuteVariant(tokens=8, hidden_size=16, experts=4, seed=0),
    RoutingMapUnpermuteVariant(tokens=256, hidden_size=64, experts=8, seed=1),
    RoutingMapUnpermuteVariant(tokens=128, hidden_size=128, experts=8, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _build_top1_routing_map(variant: RoutingMapUnpermuteVariant, generator: torch.Generator) -> torch.Tensor:
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


def _top1_routing_indices(routing_map: torch.Tensor) -> torch.Tensor:
    token_ids = torch.arange(routing_map.shape[0], dtype=torch.int64)
    sorted_indices_first = token_ids.expand(routing_map.shape[1], -1).masked_select(
        routing_map.transpose(0, 1).to(torch.bool)
    )
    return torch.argsort(sorted_indices_first).to(torch.int32)


def make_routing_map_unpermute_inputs(
    variant: RoutingMapUnpermuteVariant = VARIANT,
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
    routing_map_cpu = _build_top1_routing_map(variant, generator)
    sorted_indices_cpu = _top1_routing_indices(routing_map_cpu)
    gather_indices_cpu = (
        sorted_indices_cpu.to(torch.int64)[:, None] * variant.hidden_size
        + torch.arange(variant.hidden_size, dtype=torch.int64)[None, :]
    ).reshape(-1).to(torch.int32)
    reference_tokens_cpu = permuted_tokens_cpu.float().index_select(0, sorted_indices_cpu.to(torch.int64))

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "permuted_tokens": permuted_tokens_cpu.to(torch.float16).npu(),
        "sorted_indices": sorted_indices_cpu.npu(),
        "routing_map": routing_map_cpu.npu(),
        "gather_indices": gather_indices_cpu.npu(),
        "restored_tokens_pto": torch.empty_like(permuted_tokens_cpu).npu(),
        "reference_tokens": reference_tokens_cpu,
    }


def run_torch_npu_moe_token_unpermute_with_routing_map(inputs: dict[str, object]):
    return torch_npu.npu_moe_token_unpermute_with_routing_map(
        inputs["permuted_tokens"],
        inputs["sorted_indices"],
        inputs["shape_summary"]["restore_shape"],
        probs=None,
        routing_map=inputs["routing_map"],
        drop_and_pad=inputs["variant"]["drop_and_pad"],
    )


def run_pto_moe_token_unpermute_with_routing_map_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["restored_tokens_pto"],
        inputs["permuted_tokens"],
        inputs["gather_indices"],
    )
    return inputs["restored_tokens_pto"].float()
