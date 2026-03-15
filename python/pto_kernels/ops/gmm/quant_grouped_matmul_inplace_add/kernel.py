"""PTO kernel placeholder for quant_grouped_matmul_inplace_add."""

from __future__ import annotations


def build_jit_wrapper(*, output_dir):
    del output_dir
    raise NotImplementedError(
        "quant_grouped_matmul_inplace_add PTO port is deferred until a reproducible baseline host contract exists."
    )
