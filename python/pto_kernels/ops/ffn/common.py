"""Shared staged FFN helpers for PTO seed kernels."""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto, tile
from ptodsl import scalar as s


const = s.const


@dataclass(frozen=True)
class DenseReluFfnConfig:
    tokens: int
    hidden: int
    intermediate: int
    output: int
    base_k1: int
    base_k2: int

    @property
    def k1_iters(self) -> int:
        return self.hidden // self.base_k1

    @property
    def k2_iters(self) -> int:
        return self.intermediate // self.base_k2


def _matmul_meta_data(*, m: int, k: int, n: int, base_k: int):
    dtype = pto.float16
    acc_dtype = pto.float32

    ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)

    view_a = pto.SubTensorType(shape=[m, base_k], dtype=dtype)
    view_b = pto.SubTensorType(shape=[base_k, n], dtype=dtype)
    view_out = pto.SubTensorType(shape=[m, n], dtype=dtype)

    a_mat = pto.TileBufType(shape=[m, base_k], dtype=dtype, memory_space="MAT")
    b_mat = pto.TileBufType(shape=[base_k, n], dtype=dtype, memory_space="MAT")
    a_tile = pto.TileBufType(shape=[m, base_k], dtype=dtype, memory_space="LEFT")
    b_tile = pto.TileBufType(shape=[base_k, n], dtype=dtype, memory_space="RIGHT")
    out_acc = pto.TileBufType(shape=[m, n], dtype=acc_dtype, memory_space="ACC")

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


def _relu_meta_data(config: DenseReluFfnConfig):
    dtype = pto.float16
    ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=1, dtype=dtype)
    sub = pto.SubTensorType(shape=[config.intermediate], dtype=dtype)
    vec = pto.TileBufType(shape=[1, config.intermediate], dtype=dtype, memory_space="VEC")

    return {
        "ptr": ptr,
        "tensor": tensor,
        "sub": sub,
        "vec_in": vec,
        "vec_out": vec,
    }


def build_matmul_stage(
    *,
    config: DenseReluFfnConfig,
    output_dir,
    stage_name: str,
    input_m: int,
    input_k: int,
    input_n: int,
    base_k: int,
    input_event_id: int,
):
    @jit(
        meta_data=lambda: _matmul_meta_data(m=input_m, k=input_k, n=input_n, base_k=base_k),
        output_dir=output_dir,
        block_dim=1,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def _stage(out_ptr: "ptr", a_ptr: "ptr", b_ptr: "ptr") -> None:
        c0 = const(0)
        c1 = const(1)
        cM = const(input_m)
        cK = const(input_k)
        cN = const(input_n)
        cBaseK = const(base_k)
        cKIter = const(input_k // base_k)

        tv_a = pto.as_tensor(tensor, ptr=a_ptr, shape=[cM, cK], strides=[cK, c1])
        tv_b = pto.as_tensor(tensor, ptr=b_ptr, shape=[cK, cN], strides=[cN, c1])
        tv_out = pto.as_tensor(tensor, ptr=out_ptr, shape=[cM, cN], strides=[cN, c1])

        with pto.cube_section():
            a_mat_tile = pto.alloc_tile(a_mat)
            b_mat_tile = pto.alloc_tile(b_mat)
            a_tile_buf = pto.alloc_tile(a_tile)
            b_tile_buf = pto.alloc_tile(b_tile)
            out_acc_tile = pto.alloc_tile(out_acc)

            for i in pto.range(c0, cKIter, c1):
                k_off = i * cBaseK
                sv_a = pto.slice_view(view_a, source=tv_a, offsets=[c0, k_off], sizes=[cM, cBaseK])
                sv_b = pto.slice_view(view_b, source=tv_b, offsets=[k_off, c0], sizes=[cBaseK, cN])

                pto.load(sv_a, a_mat_tile)
                pto.load(sv_b, b_mat_tile)
                pto.record_wait_pair("LOAD", "MOV_M2L", event_id=input_event_id)
                tile.mov(a_mat_tile, a_tile_buf)
                tile.mov(b_mat_tile, b_tile_buf)
                pto.record_wait_pair("MOV_M2L", "MATMUL", event_id=input_event_id)

                pto.cond(
                    s.eq(i, c0),
                    lambda: tile.matmul(a_tile_buf, b_tile_buf, out_acc_tile),
                    lambda: tile.matmul_acc(out_acc_tile, a_tile_buf, b_tile_buf, out_acc_tile),
                )
                pto.record_wait_pair("MATMUL", "LOAD", event_id=input_event_id)

            pto.record_wait_pair("MATMUL", "STORE_ACC", event_id=input_event_id)
            sv_out = pto.slice_view(view_out, source=tv_out, offsets=[c0, c0], sizes=[cM, cN])
            pto.store(out_acc_tile, sv_out)
            pto.record_wait_pair("STORE_ACC", "MATMUL", event_id=input_event_id)

    _stage.__name__ = stage_name
    return _stage


def build_relu_stage(*, config: DenseReluFfnConfig, output_dir):
    @jit(
        meta_data=lambda: _relu_meta_data(config),
        output_dir=output_dir,
        block_dim=1,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def dense_relu_stage(hidden_ptr: "ptr") -> None:
        c0 = const(0)
        c1 = const(1)
        cIntermediate = const(config.intermediate)
        cTotal = const(config.tokens * config.intermediate)
        cTokens = const(config.tokens)

        tv_hidden = pto.as_tensor(tensor, ptr=hidden_ptr, shape=[cTotal], strides=[c1])

        with pto.vector_section():
            hidden_in = pto.alloc_tile(vec_in)
            hidden_out = pto.alloc_tile(vec_out)

            for row_idx in pto.range(c0, cTokens, c1):
                row_off = row_idx * cIntermediate
                sv_row = pto.slice_view(sub, source=tv_hidden, offsets=[row_off], sizes=[cIntermediate])
                pto.load(sv_row, hidden_in)
                tile.relu(hidden_in, hidden_out)
                pto.store(hidden_out, sv_row)

    return dense_relu_stage


class DenseReluFfnPipelineWrapper:
    def __init__(self, *, config: DenseReluFfnConfig, output_dir):
        self._config = config
        self._output_dir = Path(output_dir)
        self._stage1 = build_matmul_stage(
            config=config,
            output_dir=self._output_dir / "stage1",
            stage_name="ffn_stage1",
            input_m=config.tokens,
            input_k=config.hidden,
            input_n=config.intermediate,
            base_k=config.base_k1,
            input_event_id=0,
        )
        self._relu = build_relu_stage(config=config, output_dir=self._output_dir / "stage2_relu")
        self._stage3 = build_matmul_stage(
            config=config,
            output_dir=self._output_dir / "stage3",
            stage_name="ffn_stage3",
            input_m=config.tokens,
            input_k=config.intermediate,
            input_n=config.output,
            base_k=config.base_k2,
            input_event_id=1,
        )

    def _build(self):
        self._stage1._build()
        self._relu._build()
        self._stage3._build()

    def _artifact_paths(self):
        return (*self._stage1._artifact_paths(), *self._relu._artifact_paths(), *self._stage3._artifact_paths())

    @property
    def library_path(self):
        return None

    @property
    def output_dir(self):
        return str(self._output_dir)

    def __call__(self, out_ptr, hidden_ptr, x_ptr, w1_ptr, w2_ptr, stream_ptr=None):
        self._stage1(hidden_ptr, x_ptr, w1_ptr, stream_ptr=stream_ptr)
        self._relu(hidden_ptr, stream_ptr=stream_ptr)
        self._stage3(out_ptr, hidden_ptr, w2_ptr, stream_ptr=stream_ptr)
