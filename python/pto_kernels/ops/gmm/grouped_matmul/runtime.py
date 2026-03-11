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
    output_dtype: str = "float32"

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


VARIANT = DenseSingleWeightVariant()


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def make_dense_single_weight_inputs(*, device_index: int = 0) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(VARIANT.seed)

    x_cpu = torch.randn((VARIANT.m, VARIANT.k), generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    weight_cpu = torch.randn(
        (VARIANT.k, VARIANT.n), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)

    x = x_cpu.npu()
    weight = weight_cpu.npu()
    a3d = x.unsqueeze(0).contiguous()
    group_list = torch.tensor([VARIANT.m], dtype=torch.int64).npu()
    out_pto = torch.empty((VARIANT.batch, VARIANT.m, VARIANT.n), dtype=torch.float32).npu()
    return {
        "device": device,
        "x": x,
        "weight": weight,
        "a3d": a3d,
        "group_list": group_list,
        "out_pto": out_pto,
        "reference": torch.matmul(x_cpu.float(), weight_cpu.float()),
    }


def run_torch_npu_grouped_matmul(inputs: dict[str, object]):
    return torch_npu.npu_grouped_matmul(
        [inputs["x"]],
        [inputs["weight"]],
        group_list=inputs["group_list"],
        split_item=3,
        group_type=0,
        group_list_type=1,
        act_type=0,
    )


def run_pto_dense_variant(wrapper, inputs: dict[str, object]) -> torch.Tensor:
    wrapper(inputs["out_pto"], inputs["a3d"], inputs["weight"], VARIANT.batch)
    return inputs["out_pto"][0]
