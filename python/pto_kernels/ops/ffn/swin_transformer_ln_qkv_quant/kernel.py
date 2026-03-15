"""Bounded placeholder for SwinTransformerLnQkvQuant PTO bring-up."""

from __future__ import annotations


def build_jit_wrapper(output_dir):
    del output_dir
    raise NotImplementedError(
        "SwinTransformerLnQkvQuant PTO port is blocked pending a validated quantized "
        "layernorm -> int8 matmul -> dequant path in PTODSL/PTOAS/pto-isa."
    )
