"""Constrained PTO-DSL seed for matmul_reduce_scatter on 910B.

The phase-1 PTO seed only covers the local dense matmul portion:

1. PTO cube matmul `x1 @ x2 -> local_mm`
2. the benchmark harness performs HCCL `all_reduce(sum)` plus row chunking
   outside PTODSL to match the reduce_scatter contract

This keeps the MC2 seed runnable on the current machine while reusable PTODSL
collective primitives and true overlap-aware lowering remain tracked
separately.
"""

from dataclasses import dataclass

from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class MatmulReduceScatterConfig:
    m: int
    k: int
    n: int
    world_size: int
    rank_id: int
    base_m: int
    base_k: int

    @property
    def k_iters(self) -> int:
        return self.k // self.base_k

    @property
    def local_m(self) -> int:
        return self.m // self.world_size


def _config() -> MatmulReduceScatterConfig:
    world_size = tuned_int("PTO_MC2_WORLD_SIZE", 2, valid_values=(2, 4, 8))
    rank_id = tuned_int("PTO_MC2_RANK", 0, minimum=0, valid_values=tuple(range(world_size)))
    return MatmulReduceScatterConfig(
        m=tuned_int("PTO_MC2_M", 128, valid_values=(64, 128)),
        k=tuned_int("PTO_MC2_K", 256, valid_values=(128, 256)),
        n=tuned_int("PTO_MC2_N", 128, valid_values=(128,)),
        world_size=world_size,
        rank_id=rank_id,
        base_m=tuned_int("PTO_MC2_BASE_M", 32, valid_values=(16, 32, 64)),
        base_k=tuned_int("PTO_MC2_BASE_K", 32, valid_values=(32, 64)),
    )


def _meta_data(config: MatmulReduceScatterConfig):
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

    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=1,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def matmul_reduce_scatter_local_mm(out_ptr: "ptr", x1_ptr: "ptr", x2_ptr: "ptr") -> None:
        c0 = const(0)
        c1 = const(1)
        cM = const(config.m)
        cK = const(config.k)
        cN = const(config.n)
        cRank = const(config.rank_id)
        cLocalM = const(config.local_m)
        cBaseM = const(config.base_m)
        cBaseK = const(config.base_k)
        cIter = const(config.k_iters)
        cRowTiles = const(config.local_m // config.base_m)
        cWorld = const(config.world_size)

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

        with pto.section.cube():
            for rank_chunk_idx in range(c0, cWorld, c1):
                target_rank = (cRank + rank_chunk_idx) % cWorld
                rank_row_offset = target_rank * cLocalM
                for row_tile_idx in range(c0, cRowTiles, c1):
                    row_off = row_tile_idx * cBaseM
                    global_row_off = rank_row_offset + row_off

                    a_mat_tile = pto.alloc_tile(a_mat)
                    b_mat_tile = pto.alloc_tile(b_mat)
                    a_tile_buf = pto.alloc_tile(a_tile)
                    b_tile_buf = pto.alloc_tile(b_tile)
                    out_acc_tile = pto.alloc_tile(out_acc)

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

    return matmul_reduce_scatter_local_mm
