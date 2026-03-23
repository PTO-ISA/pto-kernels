"""Runtime helpers for the first moe_finalize_routing_v2_grad migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class MoeFinalizeRoutingV2GradVariant:
    tokens: int
    hidden_size: int
    experts: int = 4
    top_k: int = 1
    seed: int = 0
    dtype: str = "float16"
    drop_pad_mode: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"t{self.tokens}_h{self.hidden_size}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "grad_y": [self.tokens, self.hidden_size],
            "expanded_row_idx": [self.tokens * self.top_k],
            "expanded_x": [self.tokens * self.top_k, self.hidden_size],
            "scales": [self.tokens, self.top_k],
            "expert_idx": [self.tokens, self.top_k],
            "bias": [self.experts, self.hidden_size],
            "grad_expanded_x_out": [self.tokens * self.top_k, self.hidden_size],
            "grad_scales_out": [self.tokens, self.top_k],
        }


VARIANT = MoeFinalizeRoutingV2GradVariant(tokens=16, hidden_size=16, seed=0)
VARIANTS = (
    VARIANT,
    MoeFinalizeRoutingV2GradVariant(tokens=256, hidden_size=64, seed=1),
    MoeFinalizeRoutingV2GradVariant(tokens=128, hidden_size=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def baseline_available() -> bool:
    return hasattr(torch_npu, "npu_moe_finalize_routing_v2_grad") or hasattr(
        torch.ops.npu, "npu_moe_finalize_routing_v2_grad"
    )


def make_top1_finalize_routing_v2_grad_inputs(
    variant: MoeFinalizeRoutingV2GradVariant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    grad_y_cpu = torch.randn(
        (variant.tokens, variant.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    expanded_x_cpu = torch.randn(
        (variant.tokens * variant.top_k, variant.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    scales_cpu = torch.randn(
        (variant.tokens, variant.top_k),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    expert_idx_cpu = torch.randint(
        low=0,
        high=variant.experts,
        size=(variant.tokens, variant.top_k),
        generator=generator,
        dtype=torch.int32,
    )
    expanded_row_idx_cpu = torch.randperm(
        variant.tokens * variant.top_k,
        generator=generator,
        dtype=torch.int64,
    ).to(torch.int32)
    bias_cpu = torch.randn(
        (variant.experts, variant.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)

    row_ids = expanded_row_idx_cpu.view(variant.top_k, variant.tokens).transpose(0, 1)[:, 0].to(torch.int64)
    expert_ids = expert_idx_cpu[:, 0].to(torch.int64)

    grad_expanded_x_ref = torch.zeros_like(expanded_x_cpu, dtype=torch.float32)
    grad_expanded_x_ref.index_copy_(
        0,
        row_ids,
        grad_y_cpu.float() * scales_cpu[:, :1].float(),
    )
    grad_scales_ref = (
        (expanded_x_cpu.index_select(0, row_ids).float() + bias_cpu.index_select(0, expert_ids).float())
        * grad_y_cpu.float()
    ).sum(dim=1, keepdim=True)
    grad_expanded_x_ref = grad_expanded_x_ref.to(torch.float16).float()
    grad_scales_ref = grad_scales_ref.to(torch.float16).float()

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "grad_y": grad_y_cpu.npu(),
        "expanded_row_idx": expanded_row_idx_cpu.npu(),
        "expanded_x": expanded_x_cpu.npu(),
        "scales": scales_cpu.npu(),
        "expert_idx": expert_idx_cpu.npu(),
        "bias": bias_cpu.npu(),
        "grad_expanded_x_out_pto": torch.empty_like(expanded_x_cpu).npu(),
        "grad_scales_out_pto": torch.empty_like(scales_cpu).npu(),
        "reference_grad_expanded_x_out": grad_expanded_x_ref,
        "reference_grad_scales_out": grad_scales_ref,
    }


def run_torch_npu_moe_finalize_routing_v2_grad(inputs: dict[str, object]):
    if hasattr(torch_npu, "npu_moe_finalize_routing_v2_grad"):
        return torch_npu.npu_moe_finalize_routing_v2_grad(
            inputs["grad_y"],
            inputs["expanded_row_idx"],
            inputs["expanded_x"],
            inputs["scales"],
            inputs["expert_idx"],
            inputs["bias"],
            inputs["variant"]["drop_pad_mode"],
            inputs["variant"]["tokens"] * inputs["variant"]["top_k"],
            inputs["variant"]["experts"],
            0,
        )
    return torch.ops.npu.npu_moe_finalize_routing_v2_grad(
        inputs["grad_y"],
        inputs["expanded_row_idx"],
        inputs["expanded_x"],
        inputs["scales"],
        inputs["expert_idx"],
        inputs["bias"],
        inputs["variant"]["drop_pad_mode"],
        inputs["variant"]["tokens"] * inputs["variant"]["top_k"],
        inputs["variant"]["experts"],
        0,
    )


def run_pto_moe_finalize_routing_v2_grad_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["grad_expanded_x_out_pto"],
        inputs["grad_scales_out_pto"],
        inputs["grad_y"],
        inputs["expanded_row_idx"],
        inputs["expanded_x"],
        inputs["scales"],
        inputs["expert_idx"].reshape(-1),
        inputs["bias"],
    )
    return (
        inputs["grad_expanded_x_out_pto"].float(),
        inputs["grad_scales_out_pto"].float(),
    )
