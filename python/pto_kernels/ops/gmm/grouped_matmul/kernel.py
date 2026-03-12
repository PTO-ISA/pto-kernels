"""PTO-DSL grouped_matmul seed shaped after the ops-transformer tiling flow.

This keeps the current dense single-weight seed contract, but the kernel now
follows the upstream grouped_matmul block strategy more closely:
- split the output into baseM x baseN basic blocks
- distribute blocks over cores with the same diagonal-vs-row-major policy
- keep the cube pipeline shape as GM -> MAT -> L0 -> ACC -> GM

The remaining gap to the upstream AscendC kernel is the async preload /
double-buffer callback pipeline. PTO source stays sync-free and relies on
PTOAS autosync insertion.
"""

from dataclasses import dataclass
from math import gcd

from mlir.ir import BF16Type, F32Type, IntegerType
from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class GroupedMatmulConfig:
    m: int
    k: int
    n: int
    base_m: int
    base_n: int
    base_k: int
    max_block_dim: int

    @property
    def k_iters(self) -> int:
        return self.k // self.base_k

    @property
    def m_tiles(self) -> int:
        return self.m // self.base_m

    @property
    def n_tiles(self) -> int:
        return self.n // self.base_n

    @property
    def total_tiles(self) -> int:
        return self.m_tiles * self.n_tiles

    @property
    def launch_block_dim(self) -> int:
        return min(self.max_block_dim, self.total_tiles)

    def validate(self) -> None:
        for axis_name, axis, base in (
            ("m", self.m, self.base_m),
            ("k", self.k, self.base_k),
            ("n", self.n, self.base_n),
        ):
            if axis % base != 0:
                raise ValueError(f"grouped_matmul seed requires {axis_name}={axis} to be divisible by base={base}")
        if self.launch_block_dim <= 0:
            raise ValueError("grouped_matmul seed requires a positive launch_block_dim")


THRESHOLD_BLOCK_NUM = 8
THRESHOLD_DIM_M = 5


def _align_down(value: int, alignment: int) -> int:
    return value - (value % alignment)


def _lcm(lhs: int, rhs: int) -> int:
    return lhs * rhs // gcd(lhs, rhs)


def _build_tile_schedule(config: GroupedMatmulConfig) -> tuple[tuple[int, int], ...]:
    schedule: list[tuple[int, int]] = []
    total_tiles = config.total_tiles
    threshold_m_dim_n = THRESHOLD_BLOCK_NUM * config.n_tiles

    for relative_block in range(total_tiles):
        if config.m_tiles <= THRESHOLD_DIM_M or THRESHOLD_DIM_M == 1:
            m_idx = relative_block // config.n_tiles
            n_idx = relative_block % config.n_tiles
        else:
            tail_m = config.m_tiles % THRESHOLD_BLOCK_NUM
            cur_threshold_m = (
                tail_m if relative_block >= _align_down(total_tiles, threshold_m_dim_n) else THRESHOLD_BLOCK_NUM
            )
            cur_threshold_m = cur_threshold_m or THRESHOLD_BLOCK_NUM

            cur_threshold_m_threshold_n = cur_threshold_m * THRESHOLD_BLOCK_NUM
            tail_n = config.n_tiles % THRESHOLD_BLOCK_NUM
            cur_threshold_n = (
                tail_n
                if relative_block % threshold_m_dim_n
                >= _align_down(cur_threshold_m * config.n_tiles, cur_threshold_m_threshold_n)
                else THRESHOLD_BLOCK_NUM
            )
            cur_threshold_n = cur_threshold_n or THRESHOLD_BLOCK_NUM

            local_relative_block = relative_block % threshold_m_dim_n % cur_threshold_m_threshold_n
            m_idx = local_relative_block % cur_threshold_m + relative_block // threshold_m_dim_n * THRESHOLD_BLOCK_NUM
            n_idx = (
                (local_relative_block + local_relative_block // _lcm(cur_threshold_m, cur_threshold_n))
                % cur_threshold_n
                + relative_block % threshold_m_dim_n // cur_threshold_m_threshold_n * THRESHOLD_BLOCK_NUM
            )

        if not (0 <= m_idx < config.m_tiles and 0 <= n_idx < config.n_tiles):
            raise ValueError(
                f"ops-transformer schedule mapped logical block {relative_block} to invalid tile ({m_idx}, {n_idx})"
            )
        schedule.append((m_idx, n_idx))

    return tuple(schedule)


def _schedule_lookup(logical_block, schedule: tuple[tuple[int, int], ...]):
    m_idx = const(schedule[0][0])
    n_idx = const(schedule[0][1])
    for block_id, (tile_m, tile_n) in enumerate(schedule[1:], start=1):
        is_current = logical_block == const(block_id)
        m_idx = pto.select(is_current, const(tile_m), m_idx)
        n_idx = pto.select(is_current, const(tile_n), n_idx)
    return m_idx, n_idx


def _config() -> GroupedMatmulConfig:
    config = GroupedMatmulConfig(
        m=tuned_int("PTO_GROUPED_MATMUL_M", 128, valid_values=(64, 128)),
        k=tuned_int("PTO_GROUPED_MATMUL_K", 128, valid_values=(64, 128, 256)),
        n=tuned_int("PTO_GROUPED_MATMUL_N", 128, valid_values=(128, 256)),
        base_m=tuned_int("PTO_GROUPED_MATMUL_BASE_M", 16, valid_values=(16, 32, 64)),
        base_n=tuned_int("PTO_GROUPED_MATMUL_BASE_N", 64, valid_values=(64, 128)),
        base_k=tuned_int("PTO_GROUPED_MATMUL_BASE_K", 64, valid_values=(32, 64)),
        max_block_dim=tuned_int("PTO_GROUPED_MATMUL_BLOCK_DIM", 16, valid_values=(1, 2, 4, 8, 16, 20)),
    )
    config.validate()
    return config


def _meta_data(config: GroupedMatmulConfig):
    bf16 = BF16Type.get()
    f32 = F32Type.get()
    ptr_out = pto.PtrType(bf16)
    ptr_in = pto.PtrType(bf16)
    i32 = IntegerType.get_signless(32)

    tensor_in = pto.TensorType(rank=2, dtype=bf16)
    tensor_out = pto.TensorType(rank=2, dtype=bf16)

    view_a = pto.SubTensorType(shape=[config.base_m, config.base_k], dtype=bf16)
    view_b = pto.SubTensorType(shape=[config.base_k, config.base_n], dtype=bf16)
    view_out = pto.SubTensorType(shape=[config.base_m, config.base_n], dtype=bf16)

    tile_a_mat = pto.TileType(shape=[config.base_m, config.base_k], dtype=bf16, memory_space="MAT")
    tile_b_mat = pto.TileType(shape=[config.base_k, config.base_n], dtype=bf16, memory_space="MAT")
    tile_a = pto.TileType(shape=[config.base_m, config.base_k], dtype=bf16, memory_space="LEFT")
    tile_b = pto.TileType(shape=[config.base_k, config.base_n], dtype=bf16, memory_space="RIGHT")
    tile_c = pto.TileType(shape=[config.base_m, config.base_n], dtype=f32, memory_space="ACC")

    return {
        "ptr_out": ptr_out,
        "ptr_in": ptr_in,
        "i32": i32,
        "tensor_in": tensor_in,
        "tensor_out": tensor_out,
        "view_a": view_a,
        "view_b": view_b,
        "view_out": view_out,
        "tile_a_mat": tile_a_mat,
        "tile_b_mat": tile_b_mat,
        "tile_a": tile_a,
        "tile_b": tile_b,
        "tile_c": tile_c,
    }


def build_jit_wrapper(*, output_dir):
    config = _config()
    schedule = _build_tile_schedule(config)

    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=config.launch_block_dim,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def grouped_matmul_dense_bf16_bf16(
        out_ptr: "ptr_out",
        a_ptr: "ptr_in",
        b_ptr: "ptr_in",
        batch_i32: "i32",
    ) -> None:
        _ = batch_i32  # The seed variant is fixed to a single batch.
        with pto.section.cube():
            c0 = const(0)
            c1 = const(1)
            cM = const(config.m)
            cK = const(config.k)
            cN = const(config.n)
            cTileM = const(config.base_m)
            cTileN = const(config.base_n)
            cBaseK = const(config.base_k)
            cIter = const(config.k_iters)
            cTotalTiles = const(config.total_tiles)

            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())

            tv_a = pto.as_tensor(
                tensor_in,
                ptr=a_ptr,
                shape=[cM, cK],
                strides=[cK, c1],
            )
            tv_b = pto.as_tensor(
                tensor_in,
                ptr=b_ptr, shape=[cK, cN], strides=[cN, c1]
            )
            tv_out = pto.as_tensor(
                tensor_out, ptr=out_ptr, shape=[cM, cN], strides=[cN, c1]
            )

            a_mat = pto.alloc_tile(tile_a_mat)
            b_mat = pto.alloc_tile(tile_b_mat)
            a_tile = pto.alloc_tile(tile_a)
            b_tile = pto.alloc_tile(tile_b)
            c_tile = pto.alloc_tile(tile_c)

            for logical_block in range(bid, cTotalTiles, num_blocks):
                m_idx, n_idx = _schedule_lookup(logical_block, schedule)
                m_off = m_idx * cTileM
                n_off = n_idx * cTileN

                for i in range(c0, cIter, c1):
                    k_off = i * cBaseK
                    sv_a = pto.slice_view(
                        view_a,
                        source=tv_a,
                        offsets=[m_off, k_off],
                        sizes=[cTileM, cBaseK],
                    )
                    sv_b = pto.slice_view(
                        view_b,
                        source=tv_b,
                        offsets=[k_off, n_off],
                        sizes=[cBaseK, cTileN],
                    )

                    pto.load(sv_a, a_mat)
                    pto.load(sv_b, b_mat)
                    pto.mov(a_mat, a_tile)
                    pto.mov(b_mat, b_tile)

                    if i == c0:
                        pto.matmul(a_tile, b_tile, c_tile)
                    else:
                        pto.matmul_acc(c_tile, a_tile, b_tile, c_tile)

                sv_out = pto.slice_view(
                    view_out,
                    source=tv_out,
                    offsets=[m_off, n_off],
                    sizes=[cTileM, cTileN],
                )
                pto.store(c_tile, sv_out)

    return grouped_matmul_dense_bf16_bf16
