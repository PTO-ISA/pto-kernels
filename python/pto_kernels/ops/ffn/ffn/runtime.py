"""Runtime helpers for the first FFN migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class DenseReluVariant:
    tokens: int = 32
    hidden_size: int = 128
    intermediate_size: int = 256
    output_size: int = 128
    seed: int = 0
    input_scale: float = 0.125
    dtype: str = "float16"
    activation: str = "relu"
    bias: bool = False

    def as_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)

    @property
    def label(self) -> str:
        return (
            f"t{self.tokens}_h{self.hidden_size}_i{self.intermediate_size}_o{self.output_size}"
        )

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x": [self.tokens, self.hidden_size],
            "weight1": [self.hidden_size, self.intermediate_size],
            "weight2": [self.intermediate_size, self.output_size],
            "hidden": [self.tokens, self.intermediate_size],
            "output": [self.tokens, self.output_size],
        }


VARIANT = DenseReluVariant()
VARIANTS = (
    DenseReluVariant(tokens=32, hidden_size=128, intermediate_size=256, output_size=128, seed=0),
    DenseReluVariant(tokens=64, hidden_size=128, intermediate_size=256, output_size=128, seed=1),
    DenseReluVariant(tokens=32, hidden_size=128, intermediate_size=512, output_size=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def make_dense_relu_inputs(
    variant: DenseReluVariant = VARIANT,
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
    ).mul(variant.input_scale).to(torch.float16)
    weight1_cpu = torch.randn(
        (variant.hidden_size, variant.intermediate_size),
        generator=generator,
        dtype=torch.float32,
    ).mul(variant.input_scale).to(torch.float16)
    weight2_cpu = torch.randn(
        (variant.intermediate_size, variant.output_size),
        generator=generator,
        dtype=torch.float32,
    ).mul(variant.input_scale).to(torch.float16)

    x = x_cpu.npu()
    weight1 = weight1_cpu.npu()
    weight2 = weight2_cpu.npu()
    reference = (torch.relu(x @ weight1) @ weight2).float().cpu()

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "x": x,
        "weight1": weight1,
        "weight2": weight2,
        "hidden_pto": torch.empty(
            (variant.tokens, variant.intermediate_size), dtype=torch.float16
        ).npu(),
        "out_pto": torch.empty(
            (variant.tokens, variant.output_size), dtype=torch.float16
        ).npu(),
        "reference": reference,
    }


def run_torch_npu_ffn(inputs: dict[str, object]):
    return torch_npu.npu_ffn(
        inputs["x"],
        inputs["weight1"],
        inputs["weight2"],
        activation=inputs["variant"]["activation"],
    )


def run_pto_ffn_variant(wrapper, inputs: dict[str, object]) -> torch.Tensor:
    wrapper(
        inputs["out_pto"],
        inputs["hidden_pto"],
        inputs["x"],
        inputs["weight1"],
        inputs["weight2"],
    )
    return inputs["out_pto"].float()
