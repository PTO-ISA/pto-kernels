"""Runtime helpers for the first moe_init_routing migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class MoeInitRoutingVariant:
    tokens: int = 16
    hidden_size: int = 16
    experts: int = 4
    seed: int = 0
    dtype: str = "float16"
    top_k: int = 1

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"t{self.tokens}_h{self.hidden_size}_e{self.experts}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x": [self.tokens, self.hidden_size],
            "row_idx": [self.tokens, self.top_k],
            "expert_idx": [self.tokens, self.top_k],
            "expanded_x": [self.tokens * self.top_k, self.hidden_size],
            "expanded_row_idx": [self.tokens * self.top_k],
            "expanded_expert_idx": [self.tokens * self.top_k],
            "gather_indices": [self.tokens * self.top_k * self.hidden_size],
        }


VARIANT = MoeInitRoutingVariant()
VARIANTS = (
    MoeInitRoutingVariant(tokens=16, hidden_size=16, experts=4, seed=0),
    MoeInitRoutingVariant(tokens=256, hidden_size=64, experts=8, seed=1),
    MoeInitRoutingVariant(tokens=128, hidden_size=128, experts=8, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def make_moe_init_routing_inputs(
    variant: MoeInitRoutingVariant = VARIANT,
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

    # Constrained phase-2 slice: expert_idx is already grouped by expert, so the
    # kernel only needs gather/copy, not on-device routing sort.
    expert_ids = torch.arange(variant.tokens, dtype=torch.int64) % variant.experts
    order = torch.argsort(expert_ids, stable=True)
    expert_idx_cpu = expert_ids.index_select(0, order).to(torch.int32).view(variant.tokens, 1)
    row_idx_cpu = order.to(torch.int32).view(variant.tokens, 1)

    flat_rows = row_idx_cpu.view(-1).to(torch.int64)
    inverse_rows_cpu = torch.empty_like(row_idx_cpu.view(-1))
    inverse_rows_cpu[flat_rows] = torch.arange(flat_rows.numel(), dtype=torch.int32)
    gather_indices_cpu = (
        flat_rows[:, None] * variant.hidden_size
        + torch.arange(variant.hidden_size, dtype=torch.int64)[None, :]
    ).reshape(-1).to(torch.int32)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "x": x_cpu.npu(),
        "row_idx": row_idx_cpu.npu(),
        "expert_idx": expert_idx_cpu.npu(),
        "gather_indices": gather_indices_cpu.npu(),
        "expanded_x_pto": torch.empty((variant.tokens, variant.hidden_size), dtype=torch.float16).npu(),
        "expanded_row_idx_pto": torch.empty((variant.tokens,), dtype=torch.int32).npu(),
        "expanded_expert_idx_pto": torch.empty((variant.tokens,), dtype=torch.int32).npu(),
        "reference_expanded_x": x_cpu.index_select(0, flat_rows).float(),
        "reference_expanded_row_idx": inverse_rows_cpu.cpu(),
        "reference_expanded_expert_idx": expert_idx_cpu.view(-1).cpu(),
    }


def run_torch_npu_moe_init_routing(inputs: dict[str, object]):
    return torch.ops.npu.npu_moe_init_routing(
        inputs["x"],
        inputs["row_idx"],
        inputs["expert_idx"],
        int(inputs["variant"]["tokens"]),
    )


def run_pto_moe_init_routing_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["expanded_x_pto"],
        inputs["expanded_row_idx_pto"],
        inputs["expanded_expert_idx_pto"],
        inputs["x"],
        inputs["row_idx"],
        inputs["expert_idx"],
        inputs["gather_indices"],
    )
    return (
        inputs["expanded_x_pto"].float().cpu(),
        inputs["expanded_row_idx_pto"].to(torch.int32).cpu(),
        inputs["expanded_expert_idx_pto"].to(torch.int32).cpu(),
    )
