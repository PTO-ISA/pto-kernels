"""First runnable PTO-DSL slice for grouped_matmul on 910B.

This is intentionally constrained to the dense single-weight path used to bring
up the end-to-end PTO flow on the current machine:
- batch = 1
- x shape = [128, 128]
- weight shape = [128, 128]
- input dtype = bf16
- accumulator dtype = f32
- output dtype = bf16

It is not the full ops-transformer grouped_matmul semantics yet. The missing
multi-group routing and quantization are tracked in the gap board and kept out
of this seed variant on purpose.
"""

from dataclasses import dataclass

from mlir.ir import BF16Type, F32Type, IntegerType
from ptodsl import jit, pto, tile
from ptodsl import scalar as s
from pto_kernels.utils.tuning import tuned_int


const = s.const


@dataclass(frozen=True)
class GroupedMatmulConfig:
    m: int
    k: int
    n: int
    base_k: int

    @property
    def k_iters(self) -> int:
        return self.k // self.base_k


def _config() -> GroupedMatmulConfig:
    return GroupedMatmulConfig(
        m=tuned_int("PTO_GROUPED_MATMUL_M", 128, valid_values=(64, 128)),
        k=tuned_int("PTO_GROUPED_MATMUL_K", 128, valid_values=(64, 128, 256)),
        n=tuned_int("PTO_GROUPED_MATMUL_N", 128, valid_values=(128, 256)),
        base_k=tuned_int("PTO_GROUPED_MATMUL_BASE_K", 64, valid_values=(32, 64)),
    )


def _meta_data(config: GroupedMatmulConfig):
    bf16 = BF16Type.get()
    f32 = F32Type.get()
    ptr_out = pto.PtrType(bf16)
    ptr_in = pto.PtrType(bf16)
    i32 = IntegerType.get_signless(32)

    tensor_in = pto.TensorType(rank=2, dtype=bf16)
    tensor_out = pto.TensorType(rank=2, dtype=bf16)

    view_a = pto.SubTensorType(shape=[config.m, config.base_k], dtype=bf16)
    view_b = pto.SubTensorType(shape=[config.base_k, config.n], dtype=bf16)
    view_out = pto.SubTensorType(shape=[config.m, config.n], dtype=bf16)

    tile_a_mat = pto.TileBufType(shape=[config.m, config.base_k], dtype=bf16, memory_space="MAT")
    tile_b_mat = pto.TileBufType(shape=[config.base_k, config.n], dtype=bf16, memory_space="MAT")
    tile_a = pto.TileBufType(shape=[config.m, config.base_k], dtype=bf16, memory_space="LEFT")
    tile_b = pto.TileBufType(shape=[config.base_k, config.n], dtype=bf16, memory_space="RIGHT")
    tile_c = pto.TileBufType(shape=[config.m, config.n], dtype=f32, memory_space="ACC")

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

    @jit(
        meta_data=lambda: _meta_data(config),
        output_dir=output_dir,
        block_dim=1,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def grouped_matmul_dense_bf16_bf16(
        out_ptr: "ptr_out",
        a_ptr: "ptr_in",
        b_ptr: "ptr_in",
        batch_i32: "i32",
    ) -> None:
        del batch_i32  # The seed variant is fixed to a single batch.
        with pto.cube_section():
            c0 = const(0)
            c1 = const(1)
            cM = const(config.m)
            cN = const(config.n)
            cBaseK = const(config.base_k)
            cIter = const(config.k_iters)

            tv_a = pto.as_tensor(
                tensor_in,
                ptr=a_ptr,
                shape=[cM, const(config.k)],
                strides=[const(config.k), c1],
            )
            tv_b = pto.as_tensor(
                tensor_in,
                ptr=b_ptr,
                shape=[const(config.k), cN],
                strides=[cN, c1],
            )
            tv_out = pto.as_tensor(
                tensor_out, ptr=out_ptr, shape=[cM, cN], strides=[cN, c1]
            )

            a_mat = pto.alloc_tile(tile_a_mat)
            b_mat = pto.alloc_tile(tile_b_mat)
            a_tile = pto.alloc_tile(tile_a)
            b_tile = pto.alloc_tile(tile_b)
            c_tile = pto.alloc_tile(tile_c)

            for i in pto.range(c0, cIter, c1):
                k_off = i * cBaseK
                sv_a = pto.slice_view(
                    view_a,
                    source=tv_a,
                    offsets=[c0, k_off],
                    sizes=[cM, cBaseK],
                )
                sv_b = pto.slice_view(
                    view_b,
                    source=tv_b,
                    offsets=[k_off, c0],
                    sizes=[cBaseK, cN],
                )

                pto.load(sv_a, a_mat)
                pto.load(sv_b, b_mat)
                tile.mov(a_mat, a_tile)
                tile.mov(b_mat, b_tile)

                pto.cond(
                    s.eq(i, c0),
                    lambda: tile.matmul(a_tile, b_tile, c_tile),
                    lambda: tile.matmul_acc(c_tile, a_tile, b_tile, c_tile),
                )

            sv_out = pto.slice_view(
                view_out,
                source=tv_out,
                offsets=[c0, c0],
                sizes=[cM, cN],
            )
            pto.store(c_tile, sv_out)

    return grouped_matmul_dense_bf16_bf16
