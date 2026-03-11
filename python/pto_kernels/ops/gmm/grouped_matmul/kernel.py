"""First runnable PTO-DSL slice for grouped_matmul on 910B.

This is intentionally constrained to the dense single-weight path used to bring
up the end-to-end PTO flow on the current machine:
- batch = 1
- x shape = [128, 128]
- weight shape = [128, 128]
- input dtype = bf16
- accumulator/output dtype = f32

It is not the full ops-transformer grouped_matmul semantics yet. The missing
multi-group routing, quantization, and bf16 epilogue are tracked in the gap
board and kept out of this seed variant on purpose.
"""

from mlir.ir import BF16Type, F32Type, IntegerType
from ptodsl import jit, pto, tile
from ptodsl import scalar as s


const = s.const

M = 128
K = 128
N = 128
BASE_K = 32
K_ITERS = K // BASE_K


def _meta_data():
    bf16 = BF16Type.get()
    f32 = F32Type.get()
    ptr_out = pto.PtrType(f32)
    ptr_in = pto.PtrType(bf16)
    i32 = IntegerType.get_signless(32)

    tensor_in = pto.TensorType(rank=2, dtype=bf16)
    tensor_out = pto.TensorType(rank=2, dtype=f32)

    view_a = pto.SubTensorType(shape=[M, BASE_K], dtype=bf16)
    view_b = pto.SubTensorType(shape=[BASE_K, N], dtype=bf16)
    view_out = pto.SubTensorType(shape=[M, N], dtype=f32)

    tile_a_mat = pto.TileBufType(shape=[M, BASE_K], dtype=bf16, memory_space="MAT")
    tile_b_mat = pto.TileBufType(shape=[BASE_K, N], dtype=bf16, memory_space="MAT")
    tile_a = pto.TileBufType(shape=[M, BASE_K], dtype=bf16, memory_space="LEFT")
    tile_b = pto.TileBufType(shape=[BASE_K, N], dtype=bf16, memory_space="RIGHT")
    tile_c = pto.TileBufType(shape=[M, N], dtype=f32, memory_space="ACC")

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
    @jit(
        meta_data=_meta_data,
        output_dir=output_dir,
        block_dim=1,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def grouped_matmul_dense_bf16_f32(
        out_ptr: "ptr_out",
        a_ptr: "ptr_in",
        b_ptr: "ptr_in",
        batch_i32: "i32",
    ) -> None:
        del batch_i32  # The seed variant is fixed to a single batch.
        with pto.cube_section():
            c0 = const(0)
            c1 = const(1)
            cM = const(M)
            cN = const(N)
            cBaseK = const(BASE_K)
            cIter = const(K_ITERS)

            tv_a = pto.as_tensor(tensor_in, ptr=a_ptr, shape=[cM, const(K)], strides=[const(K), c1])
            tv_b = pto.as_tensor(tensor_in, ptr=b_ptr, shape=[const(K), cN], strides=[cN, c1])
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
                pto.record_wait_pair("LOAD", "MOV_M2L", event_id=0)
                tile.mov(a_mat, a_tile)
                tile.mov(b_mat, b_tile)
                pto.record_wait_pair("MOV_M2L", "MATMUL", event_id=0)

                pto.cond(
                    s.eq(i, c0),
                    lambda: tile.matmul(a_tile, b_tile, c_tile),
                    lambda: tile.matmul_acc(c_tile, a_tile, b_tile, c_tile),
                )
                pto.record_wait_pair("MATMUL", "LOAD", event_id=0)

            pto.record_wait_pair("MATMUL", "STORE_ACC", event_id=0)
            sv_out = pto.slice_view(
                view_out,
                source=tv_out,
                offsets=[c0, c0],
                sizes=[cM, cN],
            )
            pto.store(c_tile, sv_out)
            pto.record_wait_pair("STORE_ACC", "MATMUL", event_id=0)

    return grouped_matmul_dense_bf16_f32
