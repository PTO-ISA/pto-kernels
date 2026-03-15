"""Runtime helpers for the first moe_gating_top_k slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class MoeGatingTopKVariant:
    rows: int = 8
    experts: int = 16
    seed: int = 0
    dtype: str = "float16"
    top_k: int = 1
    group_count: int = 1
    k_group: int = 1
    group_select_mode: int = 0
    renorm: int = 0
    norm_type: int = 1
    out_flag: bool = False

    def as_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"r{self.rows}_e{self.experts}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x": [self.rows, self.experts],
            "y_out": [self.rows, self.top_k],
            "expert_idx_out": [self.rows, self.top_k],
            "norm_out": [self.rows, self.experts],
        }


VARIANT = MoeGatingTopKVariant()
VARIANTS = (
    MoeGatingTopKVariant(rows=8, experts=16, seed=0),
    MoeGatingTopKVariant(rows=256, experts=64, seed=1),
    MoeGatingTopKVariant(rows=128, experts=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def make_moe_gating_top_k_inputs(
    variant: MoeGatingTopKVariant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    x_cpu = torch.randn((variant.rows, variant.experts), generator=generator, dtype=torch.float32).to(torch.float16)
    # For top1 sigmoid with no grouping or bias, sigmoid is monotonic, so argmax(sigmoid(x)) == argmax(x).
    expert_idx_ref = torch.argmax(x_cpu.float(), dim=-1, keepdim=True).to(torch.int32)
    y_ref = torch.ones((variant.rows, variant.top_k), dtype=torch.float32)
    norm_out_ref = torch.zeros((variant.rows, variant.experts), dtype=torch.float32)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "x": x_cpu.npu(),
        "y_out_pto": torch.empty((variant.rows, variant.top_k), dtype=torch.float16).npu(),
        "expert_idx_out_pto": torch.empty((variant.rows, variant.top_k), dtype=torch.int32).npu(),
        "norm_out_pto": torch.zeros((variant.rows, variant.experts), dtype=torch.float32).npu(),
        "reference_y_out": y_ref,
        "reference_expert_idx_out": expert_idx_ref.cpu(),
        "reference_norm_out": norm_out_ref,
    }


def run_torch_npu_moe_gating_top_k(inputs: dict[str, object]):
    variant = inputs["variant"]
    return torch_npu.npu_moe_gating_top_k(
        inputs["x"],
        int(variant["top_k"]),
        bias=None,
        k_group=int(variant["k_group"]),
        group_count=int(variant["group_count"]),
        group_select_mode=int(variant["group_select_mode"]),
        renorm=int(variant["renorm"]),
        norm_type=int(variant["norm_type"]),
        out_flag=bool(variant["out_flag"]),
        routed_scaling_factor=1.0,
        eps=1e-20,
    )


def run_pto_moe_gating_top_k_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["y_out_pto"],
        inputs["expert_idx_out_pto"],
        inputs["x"],
    )
    return (
        inputs["y_out_pto"].float(),
        inputs["expert_idx_out_pto"].to(torch.int32),
        inputs["norm_out_pto"].float(),
    )
