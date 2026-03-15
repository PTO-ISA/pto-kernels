"""Runtime helpers for the first moe_gating_top_k_softmax_v2 slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class MoeGatingTopKSoftmaxV2Variant:
    rows: int = 8
    experts: int = 16
    seed: int = 0
    dtype: str = "float16"
    top_k: int = 1
    renorm: int = 0
    output_softmax_result_flag: bool = False

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
        }


VARIANT = MoeGatingTopKSoftmaxV2Variant()
VARIANTS = (
    MoeGatingTopKSoftmaxV2Variant(rows=8, experts=16, seed=0),
    MoeGatingTopKSoftmaxV2Variant(rows=256, experts=64, seed=1),
    MoeGatingTopKSoftmaxV2Variant(rows=128, experts=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def baseline_available() -> bool:
    return hasattr(torch_npu, "npu_moe_gating_top_k_softmax_v2") or hasattr(
        torch.ops.npu, "npu_moe_gating_top_k_softmax_v2"
    )


def make_moe_gating_top_k_softmax_v2_inputs(
    variant: MoeGatingTopKSoftmaxV2Variant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    x_cpu = torch.randn((variant.rows, variant.experts), generator=generator, dtype=torch.float32).to(torch.float16)
    softmax_cpu = torch.softmax(x_cpu.float(), dim=-1)
    y_ref, expert_idx_ref = torch.topk(softmax_cpu, k=variant.top_k, dim=-1)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "x": x_cpu.npu(),
        "y_out_pto": torch.empty((variant.rows, variant.top_k), dtype=torch.float16).npu(),
        "expert_idx_out_pto": torch.empty((variant.rows, variant.top_k), dtype=torch.int32).npu(),
        "row_idx_scratch_pto": torch.empty((variant.rows, variant.top_k), dtype=torch.int32).npu(),
        "reference_y_out": y_ref.float(),
        "reference_expert_idx_out": expert_idx_ref.to(torch.int32).cpu(),
    }


def run_torch_npu_moe_gating_top_k_softmax_v2(inputs: dict[str, object]):
    if hasattr(torch_npu, "npu_moe_gating_top_k_softmax_v2"):
        return torch_npu.npu_moe_gating_top_k_softmax_v2(
            inputs["x"],
            None,
            int(inputs["variant"]["top_k"]),
            int(inputs["variant"]["renorm"]),
            bool(inputs["variant"]["output_softmax_result_flag"]),
        )
    return torch.ops.npu.npu_moe_gating_top_k_softmax_v2(
        inputs["x"],
        None,
        int(inputs["variant"]["top_k"]),
        int(inputs["variant"]["renorm"]),
        bool(inputs["variant"]["output_softmax_result_flag"]),
    )


def run_pto_moe_gating_top_k_softmax_v2_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["y_out_pto"],
        inputs["expert_idx_out_pto"],
        inputs["row_idx_scratch_pto"],
        inputs["x"],
        inputs["probs_tmp"],
    )
    return (
        inputs["y_out_pto"].float(),
        inputs["expert_idx_out_pto"].to(torch.int32),
    )
