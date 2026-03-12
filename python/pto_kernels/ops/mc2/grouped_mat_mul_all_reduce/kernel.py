"""Constrained PTO-DSL seed for grouped_mat_mul_all_reduce on 910B.

This first MC2 grouped-matmul seed follows the upstream local cube scheduling
more closely for one dense group:

1. each rank owns a local K shard of x and weight
2. PTO computes the local grouped-matmul contribution
3. the benchmark harness performs HCCL all_reduce(sum) over the [M, N] result

The PTO cube traversal mirrors the upstream turn-based core split:
- split N into numBlocksN chunks
- derive splitM from block_dim / numBlocksN
- each core keeps a fixed nIdx and advances mIdx across turns
"""

from dataclasses import dataclass

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const
MAX_TURN_NUM = 16


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


@dataclass(frozen=True)
class GroupedMatmulAllReduceConfig:
    m: int
    k_local: int
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
        return self.core_num // self.num_blocks_n if self.core_num % self.num_blocks_n == 0 else 1

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
        return self.k_local // self.base_k

    def validate(self) -> None:
        for axis_name, axis, base in (
            ("m", self.m, self.base_m),
            ("k_local", self.k_local, self.base_k),
            ("n", self.n, self.base_n),
        ):
            if axis % base != 0:
                raise ValueError(
                    f"grouped_mat_mul_all_reduce seed requires {axis_name}={axis} divisible by {base}"
                )
        if self.core_num <= 0:
            raise ValueError("grouped_mat_mul_all_reduce seed requires positive block_dim")
        if self.active_cores <= 0:
            raise ValueError("grouped_mat_mul_all_reduce seed requires at least one active core")


def _config() -> GroupedMatmulAllReduceConfig:
    world_size = tuned_int("PTO_MC2_GMM_AR_WORLD_SIZE", 2, valid_values=(2,))
    config = GroupedMatmulAllReduceConfig(
        m=tuned_int("PTO_MC2_GMM_AR_M", 256, valid_values=(128, 256)),
        k_local=tuned_int("PTO_MC2_GMM_AR_K_LOCAL", 128, valid_values=(128,)),
        n=tuned_int("PTO_MC2_GMM_AR_N", 128, valid_values=(128,)),
        world_size=world_size,
        base_m=tuned_int("PTO_MC2_GMM_AR_BASE_M", 32, valid_values=(16, 32, 64)),
        base_n=tuned_int("PTO_MC2_GMM_AR_BASE_N", 32, valid_values=(32, 64, 128)),
        base_k=tuned_int("PTO_MC2_GMM_AR_BASE_K", 64, valid_values=(32, 64)),
        max_block_dim=tuned_int("PTO_MC2_GMM_AR_BLOCK_DIM", 4, valid_values=(1, 2, 4, 8)),
    )
    config.validate()
    return config


def _meta_data(config: GroupedMatmulAllReduceConfig):
    dtype = pto.float16
    acc_dtype = pto.float32

    ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)

    view_a = pto.SubTensorType(shape=[config.single_m, config.base_k], dtype=dtype)
    view_b = pto.SubTensorType(shape=[config.base_k, config.single_n], dtype=dtype)
    view_out = pto.SubTensorType(shape=[config.single_m, config.single_n], dtype=dtype)

    a_mat = pto.TileType(shape=[config.single_m, config.base_k], dtype=dtype, memory_space="MAT")
    b_mat = pto.TileType(shape=[config.base_k, config.single_n], dtype=dtype, memory_space="MAT")
    a_tile = pto.TileType(shape=[config.single_m, config.base_k], dtype=dtype, memory_space="LEFT")
    b_tile = pto.TileType(shape=[config.base_k, config.single_n], dtype=dtype, memory_space="RIGHT")
    out_acc = pto.TileType(shape=[config.single_m, config.single_n], dtype=acc_dtype, memory_space="ACC")

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
    def grouped_mat_mul_all_reduce_local(out_ptr: "ptr", x_ptr: "ptr", weight_ptr: "ptr") -> None:
        c0 = const(0)
        c1 = const(1)
        cM = const(config.m)
        cK = const(config.k_local)
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

        bid = pto.index_cast(pto.get_block_idx())

        tv_x = pto.as_tensor(
            tensor,
            ptr=x_ptr,
            shape=[cM, cK],
            strides=[cK, c1],
        )
        tv_weight = pto.as_tensor(
            tensor,
            ptr=weight_ptr,
            shape=[cK, cN],
            strides=[cN, c1],
        )
        tv_out = pto.as_tensor(
            tensor,
            ptr=out_ptr,
            shape=[cM, cN],
            strides=[cN, c1],
        )

        with pto.section.cube():
            if bid < cActiveCores:
                core_m_lane = bid // cNumBlocksN
                core_n_idx = bid % cNumBlocksN

                a_mat_tile = pto.alloc_tile(a_mat)
                b_mat_tile = pto.alloc_tile(b_mat)
                a_tile_buf = pto.alloc_tile(a_tile)
                b_tile_buf = pto.alloc_tile(b_tile)
                out_acc_tile = pto.alloc_tile(out_acc)

                for turn_idx in range(c0, cTotalRun, c1):
                    m_idx = turn_idx * cSplitM + core_m_lane
                    if m_idx < cNumBlocksM:
                        row_off = m_idx * cSingleM
                        col_off = core_n_idx * cSingleN
                        for k_iter in range(c0, cKIter, c1):
                            k_off = k_iter * cBaseK
                            sv_a = pto.slice_view(
                                view_a,
                                source=tv_x,
                                offsets=[row_off, k_off],
                                sizes=[cSingleM, cBaseK],
                            )
                            sv_b = pto.slice_view(
                                view_b,
                                source=tv_weight,
                                offsets=[k_off, col_off],
                                sizes=[cBaseK, cSingleN],
                            )

                            pto.load(sv_a, a_mat_tile)
                            pto.load(sv_b, b_mat_tile)
                            pto.mov(a_mat_tile, a_tile_buf)
                            pto.mov(b_mat_tile, b_tile_buf)

                            if k_iter == c0:
                                pto.matmul(a_tile_buf, b_tile_buf, out_acc_tile)
                            else:
                                pto.matmul_acc(out_acc_tile, a_tile_buf, b_tile_buf, out_acc_tile)

                        sv_out = pto.slice_view(
                            view_out,
                            source=tv_out,
                            offsets=[row_off, col_off],
                            sizes=[cSingleM, cSingleN],
                        )
                        pto.store(out_acc_tile, sv_out)

    return grouped_mat_mul_all_reduce_local
