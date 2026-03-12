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


VARIANT = DenseReluVariant()


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def make_dense_relu_inputs(*, device_index: int = 0) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(VARIANT.seed)

    x_cpu = torch.randn(
        (VARIANT.tokens, VARIANT.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).mul(VARIANT.input_scale).to(torch.float16)
    weight1_cpu = torch.randn(
        (VARIANT.hidden_size, VARIANT.intermediate_size),
        generator=generator,
        dtype=torch.float32,
    ).mul(VARIANT.input_scale).to(torch.float16)
    weight2_cpu = torch.randn(
        (VARIANT.intermediate_size, VARIANT.output_size),
        generator=generator,
        dtype=torch.float32,
    ).mul(VARIANT.input_scale).to(torch.float16)

    x = x_cpu.npu()
    weight1 = weight1_cpu.npu()
    weight2 = weight2_cpu.npu()
    reference = (torch.relu(x @ weight1) @ weight2).float().cpu()

    return {
        "device": device,
        "x": x,
        "weight1": weight1,
        "weight2": weight2,
        "hidden_pto": torch.empty(
            (VARIANT.tokens, VARIANT.intermediate_size), dtype=torch.float16
        ).npu(),
        "out_pto": torch.empty(
            (VARIANT.tokens, VARIANT.output_size), dtype=torch.float16
        ).npu(),
        "reference": reference,
    }


def run_torch_npu_ffn(inputs: dict[str, object]):
    return torch_npu.npu_ffn(
        inputs["x"],
        inputs["weight1"],
        inputs["weight2"],
        activation=VARIANT.activation,
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
