"""Runtime helpers for the constrained SwinTransformerLnQKV PTO slice."""

from dataclasses import asdict, dataclass

import torch


HEADS = 4
HEAD_DIM = 32
EPS = 1e-5


@dataclass(frozen=True)
class SwinTransformerLnQkvVariant:
    batch: int
    seq: int
    hidden: int = 128
    heads: int = HEADS
    head_dim: int = HEAD_DIM
    base_m: int = 128
    seed: int = 0
    input_scale: float = 0.125
    dtype: str = "float16"

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)

    @property
    def tokens(self) -> int:
        return self.batch * self.seq

    @property
    def qkv_hidden(self) -> int:
        return self.hidden * 3

    @property
    def label(self) -> str:
        return f"b{self.batch}_s{self.seq}_h{self.hidden}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "inputX": [self.batch, self.seq, self.hidden],
            "gamma": [self.hidden],
            "beta": [self.hidden],
            "weight": [self.hidden, self.qkv_hidden],
            "bias": [self.qkv_hidden],
            "query_output": [self.batch, self.heads, self.seq, self.head_dim],
            "key_output": [self.batch, self.heads, self.seq, self.head_dim],
            "value_output": [self.batch, self.heads, self.seq, self.head_dim],
            "flattened_tokens": self.tokens,
        }


VARIANTS = (
    SwinTransformerLnQkvVariant(batch=1, seq=64, base_m=64, seed=0),
    SwinTransformerLnQkvVariant(batch=8, seq=256, base_m=128, seed=1),
    SwinTransformerLnQkvVariant(batch=4, seq=256, base_m=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _reshape_qkv(flat: torch.Tensor, variant: SwinTransformerLnQkvVariant) -> torch.Tensor:
    return (
        flat.reshape(variant.batch, variant.seq, variant.heads, variant.head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def make_inputs(
    variant: SwinTransformerLnQkvVariant,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    x_cpu = (
        torch.randn((variant.batch, variant.seq, variant.hidden), generator=generator, dtype=torch.float32)
        .mul(variant.input_scale)
        .to(torch.float16)
    )
    gamma_cpu = (
        torch.randn((variant.hidden,), generator=generator, dtype=torch.float32)
        .mul(variant.input_scale)
        .add(1.0)
        .to(torch.float16)
    )
    beta_cpu = (
        torch.randn((variant.hidden,), generator=generator, dtype=torch.float32)
        .mul(variant.input_scale)
        .to(torch.float16)
    )
    weight_cpu = (
        torch.randn((variant.hidden, variant.qkv_hidden), generator=generator, dtype=torch.float32)
        .mul(variant.input_scale)
        .to(torch.float16)
    )
    bias_cpu = (
        torch.randn((variant.qkv_hidden,), generator=generator, dtype=torch.float32)
        .mul(variant.input_scale)
        .to(torch.float16)
    )

    x_flat = x_cpu.reshape(variant.tokens, variant.hidden)
    x32 = x_flat.float()
    gamma32 = gamma_cpu.float()
    beta32 = beta_cpu.float()
    mean = x32.mean(dim=-1, keepdim=True)
    var = x32.sub(mean).pow(2).mean(dim=-1, keepdim=True)
    ln = x32.sub(mean).mul(torch.rsqrt(var + EPS))
    ln = ln.mul(gamma32).add(beta32)
    qkv = ln.matmul(weight_cpu.float()).add(bias_cpu.float())
    q_flat, k_flat, v_flat = torch.split(qkv, variant.hidden, dim=-1)

    reference = {
        "q": _reshape_qkv(q_flat.cpu(), variant),
        "k": _reshape_qkv(k_flat.cpu(), variant),
        "v": _reshape_qkv(v_flat.cpu(), variant),
    }

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "x": x_flat.npu(),
        "gamma": gamma_cpu.npu(),
        "beta": beta_cpu.npu(),
        "weight": weight_cpu.npu(),
        "bias": bias_cpu.npu(),
        "ln_tmp": torch.empty((variant.tokens, variant.hidden), dtype=torch.float16).npu(),
        "packed_tmp": torch.empty((variant.tokens, variant.qkv_hidden), dtype=torch.float16).npu(),
        "q_out": torch.empty((variant.tokens, variant.hidden), dtype=torch.float16).npu(),
        "k_out": torch.empty((variant.tokens, variant.hidden), dtype=torch.float16).npu(),
        "v_out": torch.empty((variant.tokens, variant.hidden), dtype=torch.float16).npu(),
        "reference": reference,
    }


def baseline_available() -> bool:
    try:
        import torch_npu
    except Exception:
        return False
    return hasattr(torch_npu, "npu_swin_transformer_ln_qkv") or hasattr(
        torch.ops.npu, "npu_swin_transformer_ln_qkv"
    )


def run_pto_variant(wrapper, inputs: dict[str, object]) -> dict[str, torch.Tensor]:
    q_flat, k_flat, v_flat = wrapper(
        inputs["q_out"],
        inputs["k_out"],
        inputs["v_out"],
        inputs["packed_tmp"],
        inputs["ln_tmp"],
        inputs["x"],
        inputs["gamma"],
        inputs["beta"],
        inputs["weight"],
        inputs["bias"],
    )
    variant = SwinTransformerLnQkvVariant(**inputs["variant"])
    return {
        "q": _reshape_qkv(q_flat.float(), variant),
        "k": _reshape_qkv(k_flat.float(), variant),
        "v": _reshape_qkv(v_flat.float(), variant),
    }
