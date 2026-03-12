"""Runtime helpers for the first grouped_matmul migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class DenseSingleWeightVariant:
    batch: int = 1
    m: int = 128
    k: int = 128
    n: int = 128
    seed: int = 0
    input_dtype: str = "bfloat16"
    output_dtype: str = "bfloat16"
    baseline_weight_layout: str = "single_3d_weight_for_v5"

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"b{self.batch}_m{self.m}_k{self.k}_n{self.n}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x": [self.m, self.k],
            "weight": [self.k, self.n],
            "weight_v5": [self.batch, self.k, self.n],
            "output": [self.batch, self.m, self.n],
        }


VARIANT = DenseSingleWeightVariant()
VARIANTS = (
    DenseSingleWeightVariant(m=128, k=128, n=128, seed=0),
    DenseSingleWeightVariant(m=64, k=64, n=128, seed=1),
    DenseSingleWeightVariant(m=128, k=128, n=256, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def make_dense_single_weight_inputs(
    variant: DenseSingleWeightVariant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    x_cpu = torch.randn((variant.m, variant.k), generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    weight_cpu = torch.randn(
        (variant.k, variant.n), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    weight_cpu_v5 = weight_cpu.unsqueeze(0).contiguous()

    x = x_cpu.npu()
    weight = weight_cpu.npu()
    weight_v5 = weight_cpu_v5.npu()
    a3d = x.unsqueeze(0).contiguous()
    group_list = torch.tensor([variant.m], dtype=torch.int64).npu()
    out_pto = torch.empty((variant.batch, variant.m, variant.n), dtype=torch.bfloat16).npu()
    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "x": x,
        "weight": weight,
        "weight_v5": weight_v5,
        "a3d": a3d,
        "group_list": group_list,
        "out_pto": out_pto,
        "baseline_reference": torch.matmul(x_cpu.float(), weight_cpu.float()).to(torch.bfloat16).float(),
    }


def run_torch_npu_grouped_matmul(inputs: dict[str, object]):
    return torch_npu.npu_grouped_matmul(
        [inputs["x"]],
        [inputs["weight_v5"]],
        group_list=inputs["group_list"],
        split_item=3,
        group_type=0,
        group_list_type=1,
        act_type=0,
    )


def run_pto_dense_variant(wrapper, inputs: dict[str, object]) -> torch.Tensor:
    wrapper(inputs["out_pto"], inputs["a3d"], inputs["weight"], int(inputs["out_pto"].shape[0]))
    return inputs["out_pto"][0].float()
