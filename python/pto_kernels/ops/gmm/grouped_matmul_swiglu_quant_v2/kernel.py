"""PTO kernel placeholder for grouped_matmul_swiglu_quant_v2."""

from __future__ import annotations


def build_jit_wrapper(*, output_dir):
    del output_dir
    raise NotImplementedError(
        "grouped_matmul_swiglu_quant_v2 PTO port is deferred until the FP8/list-valued baseline contract "
        "is reproducible on this host."
    )
