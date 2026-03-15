"""Runtime helpers for the grouped_matmul_add migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class DenseGroupedMatmulAddVariant:
    m: int = 64
    k: int = 128
    n: int = 128
    seed: int = 0
    input_dtype: str = "bfloat16"
    output_dtype: str = "float32"

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"m{self.m}_k{self.k}_n{self.n}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x_t": [self.k, self.m],
            "x_pto": [self.m, self.k],
            "weight": [self.k, self.n],
            "y_init": [self.m, self.n],
            "output": [self.m, self.n],
        }


VARIANT = DenseGroupedMatmulAddVariant()
VARIANTS = (
    DenseGroupedMatmulAddVariant(m=64, k=128, n=128, seed=0),
    DenseGroupedMatmulAddVariant(m=128, k=128, n=256, seed=1),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def make_dense_grouped_matmul_add_inputs(
    variant: DenseGroupedMatmulAddVariant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    x_t_cpu = torch.randn((variant.k, variant.m), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    x_pto_cpu = x_t_cpu.transpose(0, 1).contiguous()
    weight_cpu = torch.randn((variant.k, variant.n), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    y_init_cpu = torch.randn((variant.m, variant.n), generator=generator, dtype=torch.float32)

    reference = x_pto_cpu.float() @ weight_cpu.float() + y_init_cpu.float()

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "x_t": x_t_cpu.npu(),
        "x_pto": x_pto_cpu.npu(),
        "weight": weight_cpu.npu(),
        "y_seed": y_init_cpu.npu(),
        "y_init": y_init_cpu.npu(),
        "group_list": torch.tensor([variant.k], dtype=torch.int64).npu(),
        "reference": reference,
    }


def run_torch_npu_grouped_matmul_add(inputs: dict[str, object]):
    inputs["y_init"].copy_(inputs["y_seed"])
    torch_npu.npu_grouped_matmul_add(
        inputs["y_init"],
        inputs["x_t"],
        inputs["weight"],
        inputs["group_list"],
        transpose_x=True,
        transpose_weight=False,
        group_type=2,
    )
    return inputs["y_init"]


def run_pto_variant(wrapper, inputs: dict[str, object]) -> torch.Tensor:
    return wrapper(inputs["y_init"], inputs["x_pto"], inputs["weight"]).float()
