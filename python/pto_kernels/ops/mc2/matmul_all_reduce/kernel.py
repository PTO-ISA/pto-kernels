"""Constrained PTO-DSL seed for matmul_all_reduce on 910B.

This phase-2 seed follows the upstream local cube scheduling more closely for
the validated dense fp16 path:

1. each rank owns a local `x1` shard with full-K coverage
2. PTO computes the local `x1 @ x2` contribution with split-M and N-chunk turns
3. the benchmark harness performs HCCL all_reduce(sum) over the [M, N] result

The PTO source remains explicit-sync-free and relies on PTOAS autosync.
"""

from dataclasses import dataclass

from ptodsl import jit, pto, tile
from ptodsl import scalar as s
from pto_kernels.utils.tuning import tuned_int


const = s.const
MAX_TURN_NUM = 16


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


@dataclass(frozen=True)
class MatmulAllReduceConfig:
    m: int
    k: int
    n: int
    world_size: int
    base_m: int
    base_n: int
    base_k: int
    max_block_dim: int

    @property
    def core_num(self) -> int:
        return self.max_block_dim

    @property
    def lambda_n(self) -> int:
        return _ceil_div(self.n, self.core_num * self.base_n)

    @property
    def single_n(self) -> int:
        return self.lambda_n * self.base_n

    @property
    def num_blocks_n(self) -> int:
        return _ceil_div(self.n, self.single_n)

    @property
    def split_m(self) -> int:
        if self.core_num % self.num_blocks_n == 0:
            return self.core_num // self.num_blocks_n
        return 1

    @property
    def lambda_m(self) -> int:
        return _ceil_div(self.m, MAX_TURN_NUM * self.split_m * self.base_m)

    @property
    def single_m(self) -> int:
        value = self.lambda_m * self.base_m
        return self.m if self.m < value else value

    @property
    def num_blocks_m(self) -> int:
        return _ceil_div(self.m, self.single_m)

    @property
    def total_run(self) -> int:
        return _ceil_div(self.m, self.split_m * self.single_m)

    @property
    def active_cores(self) -> int:
        return self.split_m * self.num_blocks_n

    @property
    def k_iters(self) -> int:
        return self.k // self.base_k

    def validate(self) -> None:
        for axis_name, axis, base in (
            ("m", self.m, self.base_m),
            ("k", self.k, self.base_k),
            ("n", self.n, self.base_n),
        ):
            if axis % base != 0:
                raise ValueError(f"matmul_all_reduce seed requires {axis_name}={axis} divisible by {base}")
        if self.core_num <= 0:
            raise ValueError("matmul_all_reduce seed requires positive block_dim")
        if self.active_cores <= 0:
            raise ValueError("matmul_all_reduce seed requires at least one active core")


def _config() -> MatmulAllReduceConfig:
    world_size = tuned_int("PTO_MC2_MM_AR_WORLD_SIZE", 2, valid_values=(2,))
    config = MatmulAllReduceConfig(
        m=tuned_int("PTO_MC2_MM_AR_M", 128, valid_values=(128, 256)),
        k=tuned_int("PTO_MC2_MM_AR_K", 256, valid_values=(256,)),
        n=tuned_int("PTO_MC2_MM_AR_N", 128, valid_values=(128,)),
        world_size=world_size,
        base_m=tuned_int("PTO_MC2_MM_AR_BASE_M", 32, valid_values=(16, 32, 64)),
        base_n=tuned_int("PTO_MC2_MM_AR_BASE_N", 32, valid_values=(32, 64, 128)),
        base_k=tuned_int("PTO_MC2_MM_AR_BASE_K", 64, valid_values=(32, 64)),
        max_block_dim=tuned_int("PTO_MC2_MM_AR_BLOCK_DIM", 4, valid_values=(1, 2, 4, 8)),
    )
    config.validate()
    return config


def _meta_data(config: MatmulAllReduceConfig):
    dtype = pto.float16
    acc_dtype = pto.float32

    ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)

    view_a = pto.SubTensorType(shape=[config.single_m, config.base_k], dtype=dtype)
    view_b = pto.SubTensorType(shape=[config.base_k, config.single_n], dtype=dtype)
    view_out = pto.SubTensorType(shape=[config.single_m, config.single_n], dtype=dtype)

    a_mat = pto.TileBufType(shape=[config.single_m, config.base_k], dtype=dtype, memory_space="MAT")
    b_mat = pto.TileBufType(shape=[config.base_k, config.single_n], dtype=dtype, memory_space="MAT")
    a_tile = pto.TileBufType(shape=[config.single_m, config.base_k], dtype=dtype, memory_space="LEFT")
    b_tile = pto.TileBufType(shape=[config.base_k, config.single_n], dtype=dtype, memory_space="RIGHT")
    out_acc = pto.TileBufType(shape=[config.single_m, config.single_n], dtype=acc_dtype, memory_space="ACC")

    return {
        "ptr": ptr,
        "tensor": tensor,
        "view_a": view_a,
        "view_b": view_b,
        "view_out": view_out,
        "a_mat": a_mat,
        "b_mat": b_mat,
        "a_tile": a_tile,
        "b_tile": b_tile,
        "out_acc": out_acc,
    }


def build_jit_wrapper(*, output_dir):
    config = _config()

    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=config.core_num,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def matmul_all_reduce_local(out_ptr: "ptr", x1_ptr: "ptr", x2_ptr: "ptr") -> None:
        c0 = const(0)
        c1 = const(1)
        cM = const(config.m)
        cK = const(config.k)
        cN = const(config.n)
        cSingleM = const(config.single_m)
        cSingleN = const(config.single_n)
        cBaseK = const(config.base_k)
        cKIter = const(config.k_iters)
        cTotalRun = const(config.total_run)
        cNumBlocksM = const(config.num_blocks_m)
        cNumBlocksN = const(config.num_blocks_n)
        cSplitM = const(config.split_m)
        cActiveCores = const(config.active_cores)

        bid = s.index_cast(pto.get_block_idx())

        tv_x1 = pto.as_tensor(
            tensor,
            ptr=x1_ptr,
            shape=[cM, cK],
            strides=[cK, c1],
        )
        tv_x2 = pto.as_tensor(
            tensor,
            ptr=x2_ptr,
            shape=[cK, cN],
            strides=[cN, c1],
        )
        tv_out = pto.as_tensor(
            tensor,
            ptr=out_ptr,
            shape=[cM, cN],
            strides=[cN, c1],
        )

        with pto.cube_section():
            pto.cond(
                s.lt(bid, cActiveCores),
                lambda: _run_core(
                    bid,
                    c0,
                    c1,
                    cSingleM,
                    cSingleN,
                    cBaseK,
                    cKIter,
                    cTotalRun,
                    cNumBlocksM,
                    cNumBlocksN,
                    cSplitM,
                    tv_x1,
                    tv_x2,
                    tv_out,
                ),
                lambda: None,
            )

    return matmul_all_reduce_local


def _run_core(
    bid,
    c0,
    c1,
    cSingleM,
    cSingleN,
    cBaseK,
    cKIter,
    cTotalRun,
    cNumBlocksM,
    cNumBlocksN,
    cSplitM,
    tv_x1,
    tv_x2,
    tv_out,
):
    core_m_lane = bid // cNumBlocksN
    core_n_idx = bid % cNumBlocksN

    a_mat_tile = pto.alloc_tile(a_mat)
    b_mat_tile = pto.alloc_tile(b_mat)
    a_tile_buf = pto.alloc_tile(a_tile)
    b_tile_buf = pto.alloc_tile(b_tile)
    out_acc_tile = pto.alloc_tile(out_acc)

    for turn_idx in pto.range(c0, cTotalRun, c1):
        m_idx = turn_idx * cSplitM + core_m_lane
        pto.cond(
            s.lt(m_idx, cNumBlocksM),
            lambda: _run_tile(
                m_idx,
                core_n_idx,
                c0,
                c1,
                cSingleM,
                cSingleN,
                cBaseK,
                cKIter,
                tv_x1,
                tv_x2,
                tv_out,
                a_mat_tile,
                b_mat_tile,
                a_tile_buf,
                b_tile_buf,
                out_acc_tile,
            ),
            lambda: None,
        )


def _run_tile(
    m_idx,
    n_idx,
    c0,
    c1,
    cSingleM,
    cSingleN,
    cBaseK,
    cKIter,
    tv_x1,
    tv_x2,
    tv_out,
    a_mat_tile,
    b_mat_tile,
    a_tile_buf,
    b_tile_buf,
    out_acc_tile,
):
    row_off = m_idx * cSingleM
    col_off = n_idx * cSingleN

    for k_idx in pto.range(c0, cKIter, c1):
        k_off = k_idx * cBaseK
        sv_a = pto.slice_view(
            view_a,
            source=tv_x1,
            offsets=[row_off, k_off],
            sizes=[cSingleM, cBaseK],
        )
        sv_b = pto.slice_view(
            view_b,
            source=tv_x2,
            offsets=[k_off, col_off],
            sizes=[cBaseK, cSingleN],
        )

        pto.load(sv_a, a_mat_tile)
        pto.load(sv_b, b_mat_tile)
        tile.mov(a_mat_tile, a_tile_buf)
        tile.mov(b_mat_tile, b_tile_buf)

        pto.cond(
            s.eq(k_idx, c0),
            lambda: tile.matmul(a_tile_buf, b_tile_buf, out_acc_tile),
            lambda: tile.matmul_acc(out_acc_tile, a_tile_buf, b_tile_buf, out_acc_tile),
        )

    sv_out = pto.slice_view(
        view_out,
        source=tv_out,
        offsets=[row_off, col_off],
        sizes=[cSingleM, cSingleN],
    )
    pto.store(out_acc_tile, sv_out)
