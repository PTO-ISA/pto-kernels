"""Constrained PTO-DSL seed for all_gather_matmul on 910B.

The first PTO slice follows the upstream MC2 kernel shape more closely for the
validated 2-rank dense path:

1. the benchmark harness performs a real HCCL all_gather of the local x1 shard
2. PTO computes the global gathered matmul `allgather(x1) @ x2`
3. PTO traverses gathered rank chunks in the same local-first wrapped order as
   the upstream AscendC kernel while storing results in global-rank row order

This keeps PTO source explicit-sync-free and lets PTOAS insert the required
sync edges.
"""

from dataclasses import dataclass

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class AllGatherMatmulConfig:
    global_m: int
    k: int
    n: int
    world_size: int
    rank_id: int
    base_m: int
    base_k: int
    max_block_dim: int

    @property
    def local_m(self) -> int:
        return self.global_m // self.world_size

    @property
    def row_tiles_per_rank(self) -> int:
        return self.local_m // self.base_m

    @property
    def total_tiles(self) -> int:
        return self.world_size * self.row_tiles_per_rank

    @property
    def k_iters(self) -> int:
        return self.k // self.base_k

    @property
    def launch_block_dim(self) -> int:
        return min(self.max_block_dim, self.total_tiles)

    def validate(self) -> None:
        for axis_name, axis, base in (
            ("global_m", self.global_m, self.world_size),
            ("local_m", self.local_m, self.base_m),
            ("k", self.k, self.base_k),
        ):
            if axis % base != 0:
                raise ValueError(f"all_gather_matmul seed requires {axis_name}={axis} divisible by {base}")
        if self.launch_block_dim <= 0:
            raise ValueError("all_gather_matmul seed requires a positive launch_block_dim")


def _config() -> AllGatherMatmulConfig:
    world_size = tuned_int("PTO_MC2_ALL_GATHER_WORLD_SIZE", 2, valid_values=(2,))
    rank_id = tuned_int("PTO_MC2_ALL_GATHER_RANK", 0, minimum=0, valid_values=tuple(range(world_size)))
    config = AllGatherMatmulConfig(
        global_m=tuned_int("PTO_MC2_ALL_GATHER_M", 128, valid_values=(128, 256)),
        k=tuned_int("PTO_MC2_ALL_GATHER_K", 256, valid_values=(256,)),
        n=tuned_int("PTO_MC2_ALL_GATHER_N", 128, valid_values=(128,)),
        world_size=world_size,
        rank_id=rank_id,
        base_m=tuned_int("PTO_MC2_ALL_GATHER_BASE_M", 32, valid_values=(16, 32, 64)),
        base_k=tuned_int("PTO_MC2_ALL_GATHER_BASE_K", 64, valid_values=(32, 64)),
        max_block_dim=tuned_int("PTO_MC2_ALL_GATHER_BLOCK_DIM", 4, valid_values=(1, 2, 4, 8)),
    )
    config.validate()
    return config


def _build_traversal(config: AllGatherMatmulConfig) -> tuple[tuple[int, int], ...]:
    schedule: list[tuple[int, int]] = []
    for rank_step in range(config.world_size):
        target_rank = (config.rank_id + rank_step) % config.world_size
        for row_tile_idx in range(config.row_tiles_per_rank):
            schedule.append((target_rank, row_tile_idx))
    return tuple(schedule)


def _schedule_lookup(logical_tile, schedule: tuple[tuple[int, int], ...]):
    target_rank = const(schedule[0][0])
    row_tile_idx = const(schedule[0][1])
    for tile_id, (schedule_rank, schedule_row_tile) in enumerate(schedule[1:], start=1):
        is_current = logical_tile == const(tile_id)
        target_rank = pto.select(is_current, const(schedule_rank), target_rank)
        row_tile_idx = pto.select(is_current, const(schedule_row_tile), row_tile_idx)
    return target_rank, row_tile_idx


def _meta_data(config: AllGatherMatmulConfig):
    dtype = pto.float16
    acc_dtype = pto.float32

    ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)

    view_a = pto.SubTensorType(shape=[config.base_m, config.base_k], dtype=dtype)
    view_b = pto.SubTensorType(shape=[config.base_k, config.n], dtype=dtype)
    view_out = pto.SubTensorType(shape=[config.base_m, config.n], dtype=dtype)

    a_mat = pto.TileType(shape=[config.base_m, config.base_k], dtype=dtype, memory_space="MAT")
    b_mat = pto.TileType(shape=[config.base_k, config.n], dtype=dtype, memory_space="MAT")
    a_tile = pto.TileType(shape=[config.base_m, config.base_k], dtype=dtype, memory_space="LEFT")
    b_tile = pto.TileType(shape=[config.base_k, config.n], dtype=dtype, memory_space="RIGHT")
    out_acc = pto.TileType(shape=[config.base_m, config.n], dtype=acc_dtype, memory_space="ACC")

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
    traversal = _build_traversal(config)

    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=config.launch_block_dim,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def all_gather_matmul_dense(out_ptr: "ptr", gathered_x1_ptr: "ptr", x2_ptr: "ptr") -> None:
        c0 = const(0)
        c1 = const(1)
        cM = const(config.global_m)
        cK = const(config.k)
        cN = const(config.n)
        cLocalM = const(config.local_m)
        cBaseM = const(config.base_m)
        cBaseK = const(config.base_k)
        cIter = const(config.k_iters)
        cTotalTiles = const(config.total_tiles)

        bid = pto.index_cast(pto.get_block_idx())
        num_blocks = pto.index_cast(pto.get_block_num())

        tv_x1 = pto.as_tensor(
            tensor,
            ptr=gathered_x1_ptr,
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

        with pto.section.cube():
            a_mat_tile = pto.alloc_tile(a_mat)
            b_mat_tile = pto.alloc_tile(b_mat)
            a_tile_buf = pto.alloc_tile(a_tile)
            b_tile_buf = pto.alloc_tile(b_tile)
            out_acc_tile = pto.alloc_tile(out_acc)

            for logical_tile in range(bid, cTotalTiles, num_blocks):
                target_rank, row_tile_idx = _schedule_lookup(logical_tile, traversal)
                global_row_off = target_rank * cLocalM + row_tile_idx * cBaseM

                for i in range(c0, cIter, c1):
                    k_off = i * cBaseK
                    sv_a = pto.slice_view(
                        view_a,
                        source=tv_x1,
                        offsets=[global_row_off, k_off],
                        sizes=[cBaseM, cBaseK],
                    )
                    sv_b = pto.slice_view(
                        view_b,
                        source=tv_x2,
                        offsets=[k_off, c0],
                        sizes=[cBaseK, cN],
                    )

                    pto.load(sv_a, a_mat_tile)
                    pto.load(sv_b, b_mat_tile)
                    pto.mov(a_mat_tile, a_tile_buf)
                    pto.mov(b_mat_tile, b_tile_buf)

                    if i == c0:
                        pto.matmul(a_tile_buf, b_tile_buf, out_acc_tile)
                    else:
                        pto.matmul_acc(out_acc_tile, a_tile_buf, b_tile_buf, out_acc_tile)

                sv_out = pto.slice_view(
                    view_out,
                    source=tv_out,
                    offsets=[global_row_off, c0],
                    sizes=[cBaseM, cN],
                )
                pto.store(out_acc_tile, sv_out)

    return all_gather_matmul_dense
