"""PTO kernel placeholder for grouped_matmul_swiglu_quant."""

from __future__ import annotations


def build_jit_wrapper(*, output_dir):
    del output_dir
    raise NotImplementedError(
        "grouped_matmul_swiglu_quant PTO port is deferred until the FRACTAL_NZ baseline contract "
        "is reproducible on this host."
    )
