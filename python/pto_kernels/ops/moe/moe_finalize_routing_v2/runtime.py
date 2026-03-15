"""Runtime helpers for the first moe_finalize_routing_v2 migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class MoeFinalizeRoutingV2Variant:
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
            "expanded_x": [self.tokens * self.top_k, self.hidden_size],
            "x1": [self.tokens, self.hidden_size],
            "x2": [self.tokens, self.hidden_size],
            "bias": [self.experts, self.hidden_size],
            "scales": [self.tokens, self.top_k],
            "expanded_row_idx": [self.tokens * self.top_k],
            "expert_idx": [self.tokens, self.top_k],
            "out": [self.tokens, self.hidden_size],
        }


VARIANT = MoeFinalizeRoutingV2Variant(tokens=16, hidden_size=16, seed=0)
VARIANTS = (
    VARIANT,
    MoeFinalizeRoutingV2Variant(tokens=256, hidden_size=64, seed=1),
    MoeFinalizeRoutingV2Variant(tokens=128, hidden_size=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def baseline_available() -> bool:
    return hasattr(torch_npu, "npu_moe_finalize_routing_v2") or hasattr(torch.ops.npu, "npu_moe_finalize_routing_v2")


def make_top1_finalize_routing_v2_inputs(
    variant: MoeFinalizeRoutingV2Variant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    expanded_x_cpu = torch.randn(
        (variant.tokens * variant.top_k, variant.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    x1_cpu = torch.randn(
        (variant.tokens, variant.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    x2_cpu = torch.randn(
        (variant.tokens, variant.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    bias_cpu = torch.randn(
        (variant.experts, variant.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    scales_cpu = torch.randn(
        (variant.tokens, variant.top_k),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    expanded_row_idx_cpu = torch.randperm(
        variant.tokens * variant.top_k,
        generator=generator,
        dtype=torch.int64,
    ).to(torch.int32)
    expert_idx_cpu = torch.randint(
        low=0,
        high=variant.experts,
        size=(variant.tokens, variant.top_k),
        generator=generator,
        dtype=torch.int32,
    )

    row_ids = expanded_row_idx_cpu.view(variant.top_k, variant.tokens).transpose(0, 1)[:, 0].to(torch.int64)
    expert_ids = expert_idx_cpu[:, 0].to(torch.int64)
    reference = (
        x1_cpu.float()
        + x2_cpu.float()
        + scales_cpu[:, :1].float()
        * (
            expanded_x_cpu.index_select(0, row_ids).float()
            + bias_cpu.index_select(0, expert_ids).float()
        )
    )

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "expanded_x": expanded_x_cpu.npu(),
        "x1": x1_cpu.npu(),
        "x2": x2_cpu.npu(),
        "bias": bias_cpu.npu(),
        "scales": scales_cpu.npu(),
        "expanded_row_idx": expanded_row_idx_cpu.npu(),
        "expert_idx": expert_idx_cpu.npu(),
        "out_pto": torch.empty_like(x1_cpu).npu(),
        "reference": reference,
    }


def run_torch_npu_moe_finalize_routing_v2(inputs: dict[str, object]):
    if hasattr(torch_npu, "npu_moe_finalize_routing_v2"):
        return torch_npu.npu_moe_finalize_routing_v2(
            inputs["expanded_x"],
            inputs["expanded_row_idx"],
            inputs["x1"],
            inputs["x2"],
            inputs["bias"],
            inputs["scales"],
            inputs["expert_idx"],
            inputs["variant"]["drop_pad_mode"],
        )
    return torch.ops.npu.npu_moe_finalize_routing_v2(
        inputs["expanded_x"],
        inputs["expanded_row_idx"],
        inputs["x1"],
        inputs["x2"],
        inputs["bias"],
        inputs["scales"],
        inputs["expert_idx"],
        inputs["variant"]["drop_pad_mode"],
    )


def run_pto_moe_finalize_routing_v2_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["out_pto"],
        inputs["expanded_x"],
        inputs["x1"],
        inputs["x2"],
        inputs["bias"],
        inputs["scales"],
        inputs["expanded_row_idx"],
        inputs["expert_idx"].reshape(-1),
    )
    return inputs["out_pto"].float()
