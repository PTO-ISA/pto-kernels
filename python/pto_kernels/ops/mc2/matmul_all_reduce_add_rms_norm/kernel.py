"""Constrained PTO-DSL seed for matmul_all_reduce_add_rms_norm on 910B.

The first phase-2 slice mirrors the upstream structure in two stages:

1. local dense matmul with the same splitM/numBlocksN turn loop as
   `matmul_all_reduce`
2. vector-side `add + rms_norm` over the all-reduced matmul output

The HCCL all-reduce remains host-orchestrated for now. PTO source stays
explicit-sync-free and relies on PTOAS autosync.
"""

from dataclasses import dataclass

from ptodsl import jit, pto, tile
from ptodsl import scalar as s
from pto_kernels.utils.tuning import tuned_int

from pto_kernels.ops.mc2.matmul_all_reduce.kernel import build_jit_wrapper as build_mm_jit_wrapper


const = s.const


@dataclass(frozen=True)
class AddRmsNormConfig:
    m: int
    n: int
    block_dim: int

    def validate(self) -> None:
        if self.m <= 0 or self.n <= 0:
            raise ValueError("matmul_all_reduce_add_rms_norm seed requires positive m and n")
        if self.block_dim <= 0:
            raise ValueError("matmul_all_reduce_add_rms_norm seed requires positive block_dim")


def _config() -> AddRmsNormConfig:
    config = AddRmsNormConfig(
        m=tuned_int("PTO_MC2_MM_ARN_M", 128, valid_values=(128, 256)),
        n=tuned_int("PTO_MC2_MM_ARN_N", 128, valid_values=(128,)),
        block_dim=tuned_int("PTO_MC2_MM_ARN_BLOCK_DIM", 8, valid_values=(1, 2, 4, 8, 16)),
    )
    config.validate()
    return config


def _meta_data(config: AddRmsNormConfig):
    io_dtype = pto.float16
    acc_dtype = pto.float32

    ptr = pto.PtrType(io_dtype)
    tensor = pto.TensorType(rank=2, dtype=io_dtype)
    row_view = pto.SubTensorType(shape=[1, config.n], dtype=io_dtype)

    row_tile = pto.TileBufType(
        shape=[1, config.n],
        valid_shape=[1, config.n],
        dtype=io_dtype,
        memory_space="VEC",
        config=pto.TileBufConfig(),
    )
    row_tile_acc = pto.TileBufType(
        shape=[1, config.n],
        valid_shape=[1, config.n],
        dtype=acc_dtype,
        memory_space="VEC",
        config=pto.TileBufConfig(),
    )
    scalar_tile = pto.TileBufType(
        shape=[1, config.n],
        valid_shape=[1, 1],
        dtype=acc_dtype,
        memory_space="VEC",
        config=pto.TileBufConfig(),
    )

    return {
        "ptr": ptr,
        "tensor": tensor,
        "row_view": row_view,
        "row_tile": row_tile,
        "row_tile_acc": row_tile_acc,
        "scalar_tile": scalar_tile,
    }


def build_add_rms_norm_jit_wrapper(*, output_dir):
    config = _config()

    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.m, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def add_rms_norm_stage(
        y_ptr: "ptr",
        norm_out_ptr: "ptr",
        mm_ptr: "ptr",
        residual_ptr: "ptr",
        gamma_ptr: "ptr",
        inv_n_ptr: "ptr",
        eps_ptr: "ptr",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cM = const(config.m)
        cN = const(config.n)

        tv_mm = pto.as_tensor(tensor, ptr=mm_ptr, shape=[cM, cN], strides=[cN, c1])
        tv_residual = pto.as_tensor(tensor, ptr=residual_ptr, shape=[cM, cN], strides=[cN, c1])
        tv_y = pto.as_tensor(tensor, ptr=y_ptr, shape=[cM, cN], strides=[cN, c1])
        tv_norm = pto.as_tensor(tensor, ptr=norm_out_ptr, shape=[cM, cN], strides=[cN, c1])
        tv_gamma = pto.as_tensor(tensor, ptr=gamma_ptr, shape=[c1, cN], strides=[cN, c1])
        tv_inv_n = pto.as_tensor(tensor, ptr=inv_n_ptr, shape=[c1, cN], strides=[cN, c1])
        tv_eps = pto.as_tensor(tensor, ptr=eps_ptr, shape=[c1, cN], strides=[cN, c1])

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            num_blocks = s.index_cast(pto.get_block_num())
            rows_per_core = s.ceil_div(cM, num_blocks)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, cM)

            mm_row_f16 = pto.alloc_tile(row_tile)
            residual_row_f16 = pto.alloc_tile(row_tile)
            y_row_f16 = pto.alloc_tile(row_tile)
            norm_row_f16 = pto.alloc_tile(row_tile)
            gamma_row_f16 = pto.alloc_tile(row_tile)
            inv_n_row_f16 = pto.alloc_tile(row_tile)
            eps_row_f16 = pto.alloc_tile(row_tile)

            mm_row = pto.alloc_tile(row_tile_acc)
            residual_row = pto.alloc_tile(row_tile_acc)
            y_row = pto.alloc_tile(row_tile_acc)
            sq_row = pto.alloc_tile(row_tile_acc)
            tmp_row = pto.alloc_tile(row_tile_acc)
            scale_row = pto.alloc_tile(row_tile_acc)
            refine_row = pto.alloc_tile(row_tile_acc)
            const_row = pto.alloc_tile(row_tile_acc)
            gamma_row = pto.alloc_tile(row_tile_acc)
            inv_n_row = pto.alloc_tile(row_tile_acc)
            eps_row = pto.alloc_tile(row_tile_acc)
            scalar = pto.alloc_tile(scalar_tile)

            pto.load(
                pto.slice_view(row_view, source=tv_gamma, offsets=[c0, c0], sizes=[c1, cN]),
                gamma_row_f16,
            )
            tile.cvt(gamma_row_f16, gamma_row)
            pto.load(
                pto.slice_view(row_view, source=tv_inv_n, offsets=[c0, c0], sizes=[c1, cN]),
                inv_n_row_f16,
            )
            tile.cvt(inv_n_row_f16, inv_n_row)
            pto.load(
                pto.slice_view(row_view, source=tv_eps, offsets=[c0, c0], sizes=[c1, cN]),
                eps_row_f16,
            )
            tile.cvt(eps_row_f16, eps_row)

            for row_idx in pto.range(row_start, row_end, c1):
                sv_mm = pto.slice_view(row_view, source=tv_mm, offsets=[row_idx, c0], sizes=[c1, cN])
                sv_residual = pto.slice_view(row_view, source=tv_residual, offsets=[row_idx, c0], sizes=[c1, cN])
                sv_y = pto.slice_view(row_view, source=tv_y, offsets=[row_idx, c0], sizes=[c1, cN])
                sv_norm = pto.slice_view(row_view, source=tv_norm, offsets=[row_idx, c0], sizes=[c1, cN])

                pto.load(sv_mm, mm_row_f16)
                pto.load(sv_residual, residual_row_f16)
                tile.cvt(mm_row_f16, mm_row)
                tile.cvt(residual_row_f16, residual_row)
                tile.add(mm_row, residual_row, y_row)
                tile.cvt(y_row, y_row_f16)
                pto.store(y_row_f16, sv_y)

                tile.mul(y_row, y_row, sq_row)
                tile.row_sum(sq_row, tmp_row, scalar)
                tile.row_expand(scalar, scale_row)
                tile.mul(scale_row, inv_n_row, scale_row)
                tile.add(scale_row, eps_row, scale_row)
                tile.rsqrt(scale_row, tmp_row)
                # One Newton refinement step reduces the rsqrt drift that
                # shows up in the MC2 RMSNorm parity checks on 910B.
                tile.mul(tmp_row, tmp_row, refine_row)
                tile.mul(scale_row, refine_row, refine_row)
                tile.muls(refine_row, const(0.5, dtype=pto.float32), refine_row)
                tile.expands(const(1.5, dtype=pto.float32), const_row)
                tile.sub(const_row, refine_row, refine_row)
                tile.mul(tmp_row, refine_row, tmp_row)
                tile.mul(y_row, tmp_row, sq_row)
                tile.mul(sq_row, gamma_row, sq_row)
                tile.cvt(sq_row, norm_row_f16)
                pto.store(norm_row_f16, sv_norm)

    return add_rms_norm_stage


def build_jit_wrapper(*, output_dir):
    return build_add_rms_norm_jit_wrapper(output_dir=output_dir)
