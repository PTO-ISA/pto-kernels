"""Constrained PTO-DSL seed for inplace_matmul_all_reduce_add_rms_norm on 910B.

This phase-2 slice reuses the upstream-shaped local matmul and vector epilogue
from `matmul_all_reduce_add_rms_norm`, but validates the inplace contract:
the post-add tensor is written back into the residual buffer while a separate
`norm_out` tensor is produced.

The HCCL all-reduce remains host-orchestrated for now. PTO source stays
explicit-sync-free and relies on PTOAS autosync.
"""

from pto_kernels.ops.mc2.matmul_all_reduce_add_rms_norm.kernel import (
    build_add_rms_norm_jit_wrapper as _build_add_rms_norm_jit_wrapper,
)


def build_inplace_add_rms_norm_jit_wrapper(*, output_dir):
    return _build_add_rms_norm_jit_wrapper(output_dir=output_dir)


def build_jit_wrapper(*, output_dir):
    return build_inplace_add_rms_norm_jit_wrapper(output_dir=output_dir)
