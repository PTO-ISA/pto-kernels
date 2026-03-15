"""Runtime helpers for the first moe_init_routing_v2 migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class MoeInitRoutingV2Variant:
    tokens: int = 16
    hidden_size: int = 16
    experts: int = 4
    seed: int = 0
    dtype: str = "float16"
    top_k: int = 1
    expert_tokens_num_type: int = 1
    expert_tokens_num_flag: bool = True

    def as_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"t{self.tokens}_h{self.hidden_size}_e{self.experts}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x": [self.tokens, self.hidden_size],
            "expert_idx": [self.tokens, self.top_k],
            "expanded_x": [self.tokens * self.top_k, self.hidden_size],
            "expanded_row_idx": [self.tokens * self.top_k],
            "expert_tokens_count_or_cumsum": [self.experts],
            "expert_tokens_before_capacity": [self.experts],
        }


VARIANT = MoeInitRoutingV2Variant()
VARIANTS = (
    MoeInitRoutingV2Variant(tokens=16, hidden_size=16, experts=4, seed=0),
    MoeInitRoutingV2Variant(tokens=256, hidden_size=64, experts=8, seed=1),
    MoeInitRoutingV2Variant(tokens=128, hidden_size=128, experts=8, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def make_moe_init_routing_v2_inputs(
    variant: MoeInitRoutingV2Variant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    x_cpu = torch.randn(
        (variant.tokens, variant.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)

    # Constrained phase-2 slice: expert_idx is already grouped by expert, so
    # sortedExpertIdx is the identity order and expandedX is a direct copy.
    expert_idx_cpu = (torch.arange(variant.tokens, dtype=torch.int64) % variant.experts).to(torch.int32).view(
        variant.tokens,
        1,
    )
    counts_cpu = torch.bincount(
        expert_idx_cpu.view(-1).to(torch.int64),
        minlength=variant.experts,
    ).to(torch.int32)
    cumsum_cpu = counts_cpu.cumsum(0)
    expanded_row_idx_cpu = torch.arange(variant.tokens * variant.top_k, dtype=torch.int32)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "x": x_cpu.npu(),
        "expert_idx": expert_idx_cpu.npu(),
        "expert_idx_flat": expert_idx_cpu.view(-1).npu(),
        "expanded_x_pto": torch.empty((variant.tokens, variant.hidden_size), dtype=torch.float16).npu(),
        "expanded_row_idx_pto": torch.empty((variant.tokens,), dtype=torch.int32).npu(),
        "expert_tokens_count_or_cumsum_pto": torch.empty((variant.experts,), dtype=torch.int32).npu(),
        "expert_tokens_before_capacity_pto": torch.empty((variant.experts,), dtype=torch.int32).npu(),
        "reference_expanded_x": x_cpu.float(),
        "reference_expanded_row_idx": expanded_row_idx_cpu,
        "reference_expert_tokens_count_or_cumsum": cumsum_cpu,
        "reference_expert_tokens_before_capacity": counts_cpu,
    }


def run_torch_npu_moe_init_routing_v2(inputs: dict[str, object]):
    return torch.ops.npu.npu_moe_init_routing_v2(
        inputs["x"],
        inputs["expert_idx"],
        active_num=-1,
        expert_capacity=-1,
        expert_num=int(inputs["variant"]["experts"]),
        drop_pad_mode=0,
        expert_tokens_num_type=int(inputs["variant"]["expert_tokens_num_type"]),
        expert_tokens_num_flag=bool(inputs["variant"]["expert_tokens_num_flag"]),
        quant_mode=0,
        active_expert_range=[],
        row_idx_type=0,
    )


def run_pto_moe_init_routing_v2_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["expanded_x_pto"],
        inputs["expanded_row_idx_pto"],
        inputs["expert_tokens_count_or_cumsum_pto"],
        inputs["expert_tokens_before_capacity_pto"],
        inputs["x"],
        inputs["expert_idx_flat"],
    )
    return (
        inputs["expanded_x_pto"].float().cpu(),
        inputs["expanded_row_idx_pto"].to(torch.int32).cpu(),
        inputs["expert_tokens_count_or_cumsum_pto"].to(torch.int32).cpu(),
        inputs["expert_tokens_before_capacity_pto"].to(torch.int32).cpu(),
    )
