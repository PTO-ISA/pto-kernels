"""PTO kernel placeholder for grouped_matmul_finalize_routing."""

from __future__ import annotations


def build_jit_wrapper(*, output_dir):
    del output_dir
    raise NotImplementedError(
        "grouped_matmul_finalize_routing PTO port is deferred until the baseline routed quantized "
        "contract and weight storage layout are reproducible on this host."
    )
