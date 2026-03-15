"""Runtime helpers for the bounded SwinTransformerLnQkvQuant slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass


HEADS = 4
HEAD_DIM = 32
SEQ = 64
HIDDEN = 128


@dataclass(frozen=True)
class SwinTransformerLnQkvQuantVariant:
    batch: int
    seq: int = SEQ
    hidden: int = HIDDEN
    heads: int = HEADS
    head_dim: int = HEAD_DIM
    seed: int = 0
    base_m: int = 128
    block_dim: int = 24
    ori_height: int = 8
    ori_width: int = 8
    h_win_size: int = 8
    w_win_size: int = 8

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    @property
    def qkv_hidden(self) -> int:
        return self.hidden * 3

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x": [self.batch, self.seq, self.hidden],
            "gamma": [self.hidden],
            "beta": [self.hidden],
            "weight": [self.hidden, self.qkv_hidden],
            "bias": [self.qkv_hidden],
            "quantScale": [self.hidden],
            "quantOffset": [self.hidden],
            "dequantScale": [self.qkv_hidden],
            "query_output": [self.batch, self.heads, self.seq, self.head_dim],
            "key_output": [self.batch, self.heads, self.seq, self.head_dim],
            "value_output": [self.batch, self.heads, self.seq, self.head_dim],
            "seqLength": self.seq,
            "oriHeight": self.ori_height,
            "oriWidth": self.ori_width,
            "hWinSize": self.h_win_size,
            "wWinSize": self.w_win_size,
            "tokens": self.batch * self.seq,
        }

    @property
    def label(self) -> str:
        return f"b{self.batch}_s{self.seq}_h{self.hidden}"


VARIANTS = (
    SwinTransformerLnQkvQuantVariant(batch=1, seed=0, base_m=64),
    SwinTransformerLnQkvQuantVariant(batch=8, seed=1, base_m=128),
    SwinTransformerLnQkvQuantVariant(batch=32, seed=2, base_m=128),
)


def baseline_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        import torch
    except Exception:
        return False
    if hasattr(torch.ops.npu, "npu_swin_transformer_ln_qkv_quant"):
        return True
    return False


def baseline_blocker() -> dict[str, object]:
    return {
        "status": "blocked",
        "reason": (
            "This host does not expose a Python baseline entrypoint for "
            "SwinTransformerLnQkvQuant, and the public ACLNN documentation marks the "
            "operator unsupported on Atlas A2 / 910B."
        ),
        "entrypoint": "torch_npu.npu_swin_transformer_ln_qkv_quant",
        "environment": {
            "python_entrypoint_available": baseline_available(),
            "ops_namespace_entrypoint_available": False,
            "public_a2_support": False,
        },
        "variants": [variant.as_dict() for variant in VARIANTS],
        "shape_summaries": [variant.shape_summary for variant in VARIANTS],
    }


def pto_blocker() -> dict[str, object]:
    return {
        "status": "blocked",
        "reason": (
            "A real PTO port requires a validated PTODSL/PTOAS/pto-isa path for "
            "layernorm -> quantize(int8) -> int8 cube matmul with int32 bias -> "
            "dequantize -> q/k/v split. That upstream-faithful quantized Swin path is "
            "not exposed end to end in the current PTO stack."
        ),
        "required_stack_features": [
            "ptodsl quantized layernorm and per-channel quant surface",
            "ptoas legality/emission for quantized Swin LN-QKV IR",
            "pto-isa validated A2/A3 int8 cube + dequant epilogue path",
        ],
        "variants": [variant.as_dict() for variant in VARIANTS],
        "shape_summaries": [variant.shape_summary for variant in VARIANTS],
    }
