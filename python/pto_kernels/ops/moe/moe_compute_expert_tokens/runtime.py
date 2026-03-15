"""Runtime helpers for moe_compute_expert_tokens."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class ComputeExpertTokensVariant:
    rows: int
    num_experts: int
    seed: int = 0
    dtype: str = "int32"

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"r{self.rows}_e{self.num_experts}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "sorted_experts": [self.rows],
            "out": [self.num_experts],
        }


VARIANT = ComputeExpertTokensVariant(rows=64, num_experts=8, seed=0)
VARIANTS = (
    VARIANT,
    ComputeExpertTokensVariant(rows=4096, num_experts=64, seed=1),
    ComputeExpertTokensVariant(rows=8192, num_experts=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _make_sorted_experts(variant: ComputeExpertTokensVariant) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)
    counts = torch.randint(
        low=0,
        high=max(1, variant.rows // max(1, variant.num_experts // 2) + 1),
        size=(variant.num_experts,),
        generator=generator,
        dtype=torch.int64,
    )
    total = int(counts.sum().item())
    if total == 0:
        counts[0] = variant.rows
        total = variant.rows
    scaled = torch.floor(counts.float() * (variant.rows / float(total))).to(torch.int64)
    deficit = variant.rows - int(scaled.sum().item())
    if deficit > 0:
        scaled[:deficit] += 1
    elif deficit < 0:
        for idx in range(variant.num_experts):
            take = min(int(scaled[idx].item()), -deficit)
            scaled[idx] -= take
            deficit += take
            if deficit == 0:
                break
    values = []
    for expert, count in enumerate(scaled.tolist()):
        values.extend([expert] * count)
    if len(values) < variant.rows:
        values.extend([variant.num_experts - 1] * (variant.rows - len(values)))
    values = values[:variant.rows]
    return torch.tensor(values, dtype=torch.int32)


def _reference_output(sorted_experts_cpu: torch.Tensor, num_experts: int) -> torch.Tensor:
    positions = []
    for expert in range(num_experts):
        positions.append(int(torch.searchsorted(sorted_experts_cpu, expert + 1, right=False).item()))
    return torch.tensor(positions, dtype=torch.int32)


def make_compute_expert_tokens_inputs(
    variant: ComputeExpertTokensVariant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)

    sorted_experts_cpu = _make_sorted_experts(variant)
    reference_out = _reference_output(sorted_experts_cpu, variant.num_experts)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "sorted_experts": sorted_experts_cpu.npu(),
        "out_pto": torch.empty((variant.num_experts,), dtype=torch.int32).npu(),
        "reference_out": reference_out,
    }


def run_torch_npu_moe_compute_expert_tokens(inputs: dict[str, object]):
    return torch.ops.npu.npu_moe_compute_expert_tokens(
        inputs["sorted_experts"],
        int(inputs["variant"]["num_experts"]),
    )


def run_pto_moe_compute_expert_tokens_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["out_pto"],
        inputs["sorted_experts"],
    )
    return inputs["out_pto"].cpu()
