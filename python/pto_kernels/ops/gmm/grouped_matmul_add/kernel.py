"""PTO-DSL grouped_matmul_add seed for the current 910B migration slice."""

from dataclasses import dataclass
from pathlib import Path

import torch

from mlir.ir import BF16Type, F32Type, IntegerType
from ptodsl import jit, pto
from pto_kernels.utils.tuning import tuned_int


const = pto.const


@dataclass(frozen=True)
class GroupedMatmulAddConfig:
    m: int
    k: int
    n: int
    base_m: int
    base_n: int
    base_k: int
    block_dim: int

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
        return min(self.block_dim, self.total_tiles)

    def validate(self) -> None:
        for axis_name, axis, base in (
            ("m", self.m, self.base_m),
            ("k", self.k, self.base_k),
            ("n", self.n, self.base_n),
        ):
            if axis % base != 0:
                raise ValueError(f"grouped_matmul_add requires {axis_name}={axis} divisible by base={base}")
        if self.launch_block_dim <= 0:
            raise ValueError("grouped_matmul_add requires a positive launch block dim")


def _config() -> GroupedMatmulAddConfig:
    config = GroupedMatmulAddConfig(
        m=tuned_int("PTO_GROUPED_MATMUL_ADD_M", 64, valid_values=(64, 128)),
        k=tuned_int("PTO_GROUPED_MATMUL_ADD_K", 128, valid_values=(128,)),
        n=tuned_int("PTO_GROUPED_MATMUL_ADD_N", 128, valid_values=(128, 256)),
        base_m=tuned_int("PTO_GROUPED_MATMUL_ADD_BASE_M", 16, valid_values=(16, 32, 64)),
        base_n=tuned_int("PTO_GROUPED_MATMUL_ADD_BASE_N", 64, valid_values=(64, 128)),
        base_k=tuned_int("PTO_GROUPED_MATMUL_ADD_BASE_K", 64, valid_values=(32, 64)),
        block_dim=tuned_int("PTO_GROUPED_MATMUL_ADD_BLOCK_DIM", 16, valid_values=(1, 2, 4, 8, 16)),
    )
    config.validate()
    return config


def _matmul_meta(config: GroupedMatmulAddConfig):
    bf16 = BF16Type.get()
    f32 = F32Type.get()
    return {
        "ptr_out": pto.PtrType(f32),
        "ptr_in": pto.PtrType(bf16),
        "i32": IntegerType.get_signless(32),
        "tensor_in": pto.TensorType(rank=2, dtype=bf16),
        "tensor_out": pto.TensorType(rank=2, dtype=f32),
        "view_a": pto.SubTensorType(shape=[config.base_m, config.base_k], dtype=bf16),
        "view_b": pto.SubTensorType(shape=[config.base_k, config.base_n], dtype=bf16),
        "view_out": pto.SubTensorType(shape=[config.base_m, config.base_n], dtype=f32),
        "tile_a_mat": pto.TileType(shape=[config.base_m, config.base_k], dtype=bf16, memory_space="MAT"),
        "tile_b_mat": pto.TileType(shape=[config.base_k, config.base_n], dtype=bf16, memory_space="MAT"),
        "tile_a": pto.TileType(shape=[config.base_m, config.base_k], dtype=bf16, memory_space="LEFT"),
        "tile_b": pto.TileType(shape=[config.base_k, config.base_n], dtype=bf16, memory_space="RIGHT"),
        "tile_c": pto.TileType(shape=[config.base_m, config.base_n], dtype=f32, memory_space="ACC"),
    }


def _add_meta(config: GroupedMatmulAddConfig):
    f32 = F32Type.get()
    return {
        "ptr": pto.PtrType(f32),
        "i32": IntegerType.get_signless(32),
        "tensor": pto.TensorType(rank=2, dtype=f32),
        "row_view": pto.SubTensorType(shape=[1, config.n], dtype=f32),
        "row_tile": pto.TileType(
            shape=[1, config.n],
            valid_shape=[1, config.n],
            dtype=f32,
            memory_space="VEC",
            config=pto.TileConfig(),
        ),
    }


def _build_matmul_stage(config: GroupedMatmulAddConfig, *, output_dir):
    @jit(
        meta_data=lambda: _matmul_meta(config),
        output_dir=output_dir,
        block_dim=config.launch_block_dim,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def grouped_matmul_add_matmul_stage(
        out_ptr: "ptr_out",
        a_ptr: "ptr_in",
        b_ptr: "ptr_in",
        batch_i32: "i32",
    ) -> None:
        _ = batch_i32
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

            tv_a = pto.as_tensor(tensor_in, ptr=a_ptr, shape=[cM, cK], strides=[cK, c1])
            tv_b = pto.as_tensor(tensor_in, ptr=b_ptr, shape=[cK, cN], strides=[cN, c1])
            tv_out = pto.as_tensor(tensor_out, ptr=out_ptr, shape=[cM, cN], strides=[cN, c1])

            a_mat = pto.alloc_tile(tile_a_mat)
            b_mat = pto.alloc_tile(tile_b_mat)
            a_tile = pto.alloc_tile(tile_a)
            b_tile = pto.alloc_tile(tile_b)
            c_tile = pto.alloc_tile(tile_c)

            for logical_block in range(bid, cTotalTiles, num_blocks):
                m_idx = logical_block // const(config.n_tiles)
                n_idx = logical_block % const(config.n_tiles)
                m_off = m_idx * cTileM
                n_off = n_idx * cTileN

                for i in range(c0, cIter, c1):
                    k_off = i * cBaseK
                    sv_a = pto.slice_view(view_a, source=tv_a, offsets=[m_off, k_off], sizes=[cTileM, cBaseK])
                    sv_b = pto.slice_view(view_b, source=tv_b, offsets=[k_off, n_off], sizes=[cBaseK, cTileN])
                    pto.load(sv_a, a_mat)
                    pto.load(sv_b, b_mat)
                    pto.mov(a_mat, a_tile)
                    pto.mov(b_mat, b_tile)
                    if i == c0:
                        pto.matmul(a_tile, b_tile, c_tile)
                    else:
                        pto.matmul_acc(c_tile, a_tile, b_tile, c_tile)

                sv_out = pto.slice_view(view_out, source=tv_out, offsets=[m_off, n_off], sizes=[cTileM, cTileN])
                pto.store(c_tile, sv_out)

    return grouped_matmul_add_matmul_stage


def _build_add_stage(config: GroupedMatmulAddConfig, *, output_dir):
    @jit(
        meta_data=lambda: _add_meta(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.m, config.block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def grouped_matmul_add_add_stage(
        out_ptr: "ptr",
        mm_ptr: "ptr",
        y_ptr: "ptr",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cM = const(config.m)
        cN = const(config.n)

        tv_out = pto.as_tensor(tensor, ptr=out_ptr, shape=[cM, cN], strides=[cN, c1])
        tv_mm = pto.as_tensor(tensor, ptr=mm_ptr, shape=[cM, cN], strides=[cN, c1])
        tv_y = pto.as_tensor(tensor, ptr=y_ptr, shape=[cM, cN], strides=[cN, c1])

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cM, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cM)

            mm_row = pto.alloc_tile(row_tile)
            y_row = pto.alloc_tile(row_tile)
            out_row = pto.alloc_tile(row_tile)

            for row_idx in range(row_start, row_end, c1):
                sv_mm = pto.slice_view(row_view, source=tv_mm, offsets=[row_idx, c0], sizes=[c1, cN])
                sv_y = pto.slice_view(row_view, source=tv_y, offsets=[row_idx, c0], sizes=[c1, cN])
                sv_out = pto.slice_view(row_view, source=tv_out, offsets=[row_idx, c0], sizes=[c1, cN])
                pto.load(sv_mm, mm_row)
                pto.load(sv_y, y_row)
                pto.add(mm_row, y_row, out_row)
                pto.store(out_row, sv_out)

    return grouped_matmul_add_add_stage


class GroupedMatmulAddWrapper:
    def __init__(self, *, output_dir):
        self._output_dir = Path(output_dir)
        self._config = _config()
        self._matmul = _build_matmul_stage(self._config, output_dir=self._output_dir / "stage_matmul")
        self._add = _build_add_stage(self._config, output_dir=self._output_dir / "stage_add")

    def _build(self):
        self._matmul._build()
        self._add._build()

    def _artifact_paths(self):
        return tuple(self._matmul._artifact_paths()) + tuple(self._add._artifact_paths())

    def __call__(self, y_init, x_pto, weight, stream_ptr=None):
        mm = torch.empty((self._config.m, self._config.n), dtype=torch.float32, device=y_init.device)
        out = torch.empty_like(y_init)
        self._matmul(mm, x_pto, weight, 1, stream_ptr=stream_ptr)
        self._add(out, mm, y_init, stream_ptr=stream_ptr)
        return out


def build_jit_wrapper(*, output_dir):
    return GroupedMatmulAddWrapper(output_dir=output_dir)
