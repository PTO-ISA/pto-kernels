"""Shared staged FFN helpers for PTO seed kernels."""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto


const = pto.const


@dataclass(frozen=True)
class DenseReluFfnConfig:
    tokens: int
    hidden: int
    intermediate: int
    output: int
    base_m1: int
    base_n1: int
    base_k1: int
    block_dim1: int
    base_m2: int
    base_n2: int
    base_k2: int
    block_dim2: int
    relu_block_dim: int

    @property
    def k1_iters(self) -> int:
        return self.hidden // self.base_k1

    @property
    def k2_iters(self) -> int:
        return self.intermediate // self.base_k2

    def validate(self) -> None:
        dims = (
            ("tokens", self.tokens, self.base_m1),
            ("intermediate", self.intermediate, self.base_n1),
            ("hidden", self.hidden, self.base_k1),
            ("tokens", self.tokens, self.base_m2),
            ("output", self.output, self.base_n2),
            ("intermediate", self.intermediate, self.base_k2),
        )
        for axis_name, axis, base in dims:
            if axis % base != 0:
                raise ValueError(f"FFN seed requires {axis_name}={axis} to be divisible by base={base}")


def _launch_block_dim(total_tiles: int, requested: int) -> int:
    return max(1, min(total_tiles, requested))


def _matmul_meta_data(*, base_m: int, base_k: int, base_n: int):
    dtype = pto.float16
    acc_dtype = pto.float32

    ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)

    view_a = pto.SubTensorType(shape=[base_m, base_k], dtype=dtype)
    view_b = pto.SubTensorType(shape=[base_k, base_n], dtype=dtype)
    view_out = pto.SubTensorType(shape=[base_m, base_n], dtype=dtype)

    a_mat = pto.TileType(shape=[base_m, base_k], dtype=dtype, memory_space="MAT")
    b_mat = pto.TileType(shape=[base_k, base_n], dtype=dtype, memory_space="MAT")
    a_tile = pto.TileType(shape=[base_m, base_k], dtype=dtype, memory_space="LEFT")
    b_tile = pto.TileType(shape=[base_k, base_n], dtype=dtype, memory_space="RIGHT")
    out_acc = pto.TileType(shape=[base_m, base_n], dtype=acc_dtype, memory_space="ACC")

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


def _matmul_tile_schedule(total_m_tiles: int, total_n_tiles: int):
    return tuple(
        (m_idx, n_idx) for m_idx in range(total_m_tiles) for n_idx in range(total_n_tiles)
    )


def _schedule_lookup(logical_block, schedule):
    m_idx = const(schedule[0][0])
    n_idx = const(schedule[0][1])
    for block_id, (tile_m, tile_n) in enumerate(schedule[1:], start=1):
        is_current = logical_block == const(block_id)
        m_idx = pto.select(is_current, const(tile_m), m_idx)
        n_idx = pto.select(is_current, const(tile_n), n_idx)
    return m_idx, n_idx


def _relu_meta_data(config: DenseReluFfnConfig):
    dtype = pto.float16
    ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)
    sub = pto.SubTensorType(shape=[1, config.intermediate], dtype=dtype)
    vec = pto.TileType(shape=[1, config.intermediate], dtype=dtype, memory_space="VEC")

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
    base_m: int,
    base_n: int,
    base_k: int,
    stage_block_dim: int,
):
    schedule = _matmul_tile_schedule(input_m // base_m, input_n // base_n)

    @jit(
        meta_data=lambda: _matmul_meta_data(base_m=base_m, base_k=base_k, base_n=base_n),
        output_dir=output_dir,
        block_dim=_launch_block_dim(len(schedule), stage_block_dim),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def _stage(out_ptr: "ptr", a_ptr: "ptr", b_ptr: "ptr") -> None:
        c0 = const(0)
        c1 = const(1)
        cM = const(input_m)
        cK = const(input_k)
        cN = const(input_n)
        cBaseM = const(base_m)
        cBaseN = const(base_n)
        cBaseK = const(base_k)
        cKIter = const(input_k // base_k)
        cTotalTiles = const(len(schedule))

        tv_a = pto.as_tensor(tensor, ptr=a_ptr, shape=[cM, cK], strides=[cK, c1])
        tv_b = pto.as_tensor(tensor, ptr=b_ptr, shape=[cK, cN], strides=[cN, c1])
        tv_out = pto.as_tensor(tensor, ptr=out_ptr, shape=[cM, cN], strides=[cN, c1])

        with pto.section.cube():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            a_mat_tile = pto.alloc_tile(a_mat)
            b_mat_tile = pto.alloc_tile(b_mat)
            a_tile_buf = pto.alloc_tile(a_tile)
            b_tile_buf = pto.alloc_tile(b_tile)
            out_acc_tile = pto.alloc_tile(out_acc)

            for logical_block in range(bid, cTotalTiles, num_blocks):
                m_idx, n_idx = _schedule_lookup(logical_block, schedule)
                m_off = m_idx * cBaseM
                n_off = n_idx * cBaseN

                for i in range(c0, cKIter, c1):
                    k_off = i * cBaseK
                    sv_a = pto.slice_view(
                        view_a, source=tv_a, offsets=[m_off, k_off], sizes=[cBaseM, cBaseK]
                    )
                    sv_b = pto.slice_view(
                        view_b, source=tv_b, offsets=[k_off, n_off], sizes=[cBaseK, cBaseN]
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
                    view_out, source=tv_out, offsets=[m_off, n_off], sizes=[cBaseM, cBaseN]
                )
                pto.store(out_acc_tile, sv_out)

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
        cTokens = const(config.tokens)

        tv_hidden = pto.as_tensor(
            tensor, ptr=hidden_ptr, shape=[cTokens, cIntermediate], strides=[cIntermediate, c1]
        )

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cTokens, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cTokens)
            hidden_in = pto.alloc_tile(vec_in)
            hidden_out = pto.alloc_tile(vec_out)

            for row_idx in range(row_start, row_end, c1):
                sv_row = pto.slice_view(sub, source=tv_hidden, offsets=[row_idx, c0], sizes=[c1, cIntermediate])
                pto.load(sv_row, hidden_in)
                pto.relu(hidden_in, hidden_out)
                pto.store(hidden_out, sv_row)

    return dense_relu_stage


class DenseReluFfnPipelineWrapper:
    def __init__(self, *, config: DenseReluFfnConfig, output_dir):
        config.validate()
        self._config = config
        self._output_dir = Path(output_dir)
        self._stage1 = build_matmul_stage(
            config=config,
            output_dir=self._output_dir / "stage1",
            stage_name="ffn_stage1",
            input_m=config.tokens,
            input_k=config.hidden,
            input_n=config.intermediate,
            base_m=config.base_m1,
            base_n=config.base_n1,
            base_k=config.base_k1,
            stage_block_dim=config.block_dim1,
        )
        self._relu = build_relu_stage(config=config, output_dir=self._output_dir / "stage2_relu")
        self._relu.set_block_dim(max(1, min(config.tokens, config.relu_block_dim)))
        self._stage3 = build_matmul_stage(
            config=config,
            output_dir=self._output_dir / "stage3",
            stage_name="ffn_stage3",
            input_m=config.tokens,
            input_k=config.intermediate,
            input_n=config.output,
            base_m=config.base_m2,
            base_n=config.base_n2,
            base_k=config.base_k2,
            stage_block_dim=config.block_dim2,
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
