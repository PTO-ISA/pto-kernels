"""Runtime helpers for the first moe_gating_top_k_softmax slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class MoeGatingTopKSoftmaxVariant:
    rows: int = 8
    experts: int = 16
    seed: int = 0
    dtype: str = "float16"
    top_k: int = 1
    finished: bool = False

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
            "row_idx_out": [self.rows, self.top_k],
        }


VARIANT = MoeGatingTopKSoftmaxVariant()
VARIANTS = (
    MoeGatingTopKSoftmaxVariant(rows=8, experts=16, seed=0),
    MoeGatingTopKSoftmaxVariant(rows=256, experts=64, seed=1),
    MoeGatingTopKSoftmaxVariant(rows=128, experts=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def make_moe_gating_top_k_softmax_inputs(
    variant: MoeGatingTopKSoftmaxVariant = VARIANT,
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
    row_idx_ref = torch.arange(variant.rows, dtype=torch.int32)[:, None].expand(variant.rows, variant.top_k).contiguous()

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "x": x_cpu.npu(),
        "y_out_pto": torch.empty((variant.rows, variant.top_k), dtype=torch.float16).npu(),
        "expert_idx_out_pto": torch.empty((variant.rows, variant.top_k), dtype=torch.int32).npu(),
        "row_idx_out_pto": torch.empty((variant.rows, variant.top_k), dtype=torch.int32).npu(),
        "reference_y_out": y_ref.float(),
        "reference_expert_idx_out": expert_idx_ref.to(torch.int32).cpu(),
        "reference_row_idx_out": row_idx_ref.cpu(),
    }


def run_torch_npu_moe_gating_top_k_softmax(inputs: dict[str, object]):
    return torch_npu.npu_moe_gating_top_k_softmax(
        inputs["x"],
        None,
        int(inputs["variant"]["top_k"]),
    )


def run_pto_moe_gating_top_k_softmax_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["y_out_pto"],
        inputs["expert_idx_out_pto"],
        inputs["row_idx_out_pto"],
        inputs["x"],
        inputs["probs_tmp"],
    )
    return (
        inputs["y_out_pto"].float(),
        inputs["expert_idx_out_pto"].to(torch.int32),
        inputs["row_idx_out_pto"].to(torch.int32),
    )
