"""Runtime helpers for the first moe_token_unpermute_with_routing_map_grad slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class RoutingMapUnpermuteGradVariant:
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
            "unpermuted_tokens_grad": [self.tokens, self.hidden_size],
            "routing_map": [self.tokens, self.experts],
            "out_index": [self.tokens],
            "permute_token_id": [self.tokens],
            "restore_shape": [self.tokens, self.hidden_size],
            "permuted_tokens_grad": [self.tokens, self.hidden_size],
        }


VARIANT = RoutingMapUnpermuteGradVariant()
VARIANTS = (
    RoutingMapUnpermuteGradVariant(tokens=8, hidden_size=16, experts=4, seed=0),
    RoutingMapUnpermuteGradVariant(tokens=256, hidden_size=64, experts=8, seed=1),
    RoutingMapUnpermuteGradVariant(tokens=128, hidden_size=128, experts=8, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _build_top1_routing_map(variant: RoutingMapUnpermuteGradVariant, generator: torch.Generator) -> torch.Tensor:
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


def make_routing_map_unpermute_grad_inputs(
    variant: RoutingMapUnpermuteGradVariant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    unpermuted_grad_cpu = torch.randn(
        (variant.tokens, variant.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    routing_map_cpu = _build_top1_routing_map(variant, generator)
    out_index_cpu = _top1_routing_indices(routing_map_cpu)
    permute_token_id_cpu = torch.arange(variant.tokens, dtype=torch.int32)
    reference_permuted_grad_cpu = torch.empty_like(unpermuted_grad_cpu, dtype=torch.float32)
    reference_permuted_grad_cpu[out_index_cpu.to(torch.int64)] = unpermuted_grad_cpu.float()

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "unpermuted_tokens_grad": unpermuted_grad_cpu.npu(),
        "routing_map": routing_map_cpu.npu(),
        "out_index": out_index_cpu.npu(),
        "permute_token_id": permute_token_id_cpu.npu(),
        "permuted_tokens_grad_pto": torch.empty_like(unpermuted_grad_cpu).npu(),
        "probs_grad_pto": torch.zeros((), dtype=torch.float16).npu(),
        "reference_permuted_tokens_grad": reference_permuted_grad_cpu,
        "reference_probs_grad": torch.tensor(0.0, dtype=torch.float32),
    }


def run_torch_npu_moe_token_unpermute_with_routing_map_grad(inputs: dict[str, object]):
    return torch_npu.npu_moe_token_unpermute_with_routing_map_grad(
        inputs["unpermuted_tokens_grad"],
        inputs["out_index"],
        inputs["permute_token_id"],
        inputs["routing_map"],
        None,
        None,
        inputs["variant"]["drop_and_pad"],
        inputs["shape_summary"]["restore_shape"],
    )


def run_pto_moe_token_unpermute_with_routing_map_grad_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["permuted_tokens_grad_pto"],
        inputs["probs_grad_pto"],
        inputs["unpermuted_tokens_grad"],
        inputs["out_index"],
    )
    return inputs["permuted_tokens_grad_pto"].float(), inputs["probs_grad_pto"].float()
