"""Shared staged dense-attention helpers for PTO seed kernels."""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto


const = pto.const


@dataclass(frozen=True)
class DenseAttentionConfig:
    seq_len: int
    head_dim: int
    scores_dim: int
    qk_base_m: int
    qk_base_n: int
    qk_base_k: int
    qk_block_dim: int
    pv_base_m: int
    pv_base_n: int
    pv_base_k: int
    pv_block_dim: int
    softmax_block_dim: int
    score_scale: float = 1.0
    softmax_fp32: bool = False

    @property
    def qk_iters(self) -> int:
        return self.head_dim // self.qk_base_k

    @property
    def pv_iters(self) -> int:
        return self.scores_dim // self.pv_base_k

    def validate(self) -> None:
        dims = (
            ("seq_len", self.seq_len, self.qk_base_m),
            ("scores_dim", self.scores_dim, self.qk_base_n),
            ("head_dim", self.head_dim, self.qk_base_k),
            ("seq_len", self.seq_len, self.pv_base_m),
            ("head_dim", self.head_dim, self.pv_base_n),
            ("scores_dim", self.scores_dim, self.pv_base_k),
        )
        for axis_name, axis, base in dims:
            if axis % base != 0:
                raise ValueError(
                    f"dense attention seed requires {axis_name}={axis} to be divisible by base={base}"
                )


def _launch_block_dim(total_tiles: int, requested: int) -> int:
    return max(1, min(total_tiles, requested))


def _row_major_tile_indices(logical_block, total_n_tiles):
    return logical_block // total_n_tiles, logical_block % total_n_tiles


def _qk_meta_data(*, base_m: int, base_k: int, base_n: int):
    dtype = pto.float16
    acc_dtype = pto.float32

    scores_ptr = pto.PtrType(dtype)
    query_ptr = pto.PtrType(dtype)
    key_t_ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)

    view_q = pto.SubTensorType(shape=[base_m, base_k], dtype=dtype)
    view_kt = pto.SubTensorType(shape=[base_k, base_n], dtype=dtype)
    view_scores = pto.SubTensorType(shape=[base_m, base_n], dtype=dtype)

    q_mat = pto.TileType(shape=[base_m, base_k], dtype=dtype, memory_space="MAT")
    kt_mat = pto.TileType(shape=[base_k, base_n], dtype=dtype, memory_space="MAT")
    q_tile = pto.TileType(shape=[base_m, base_k], dtype=dtype, memory_space="LEFT")
    kt_tile = pto.TileType(shape=[base_k, base_n], dtype=dtype, memory_space="RIGHT")
    scores_acc = pto.TileType(shape=[base_m, base_n], dtype=acc_dtype, memory_space="ACC")

    return {
        "scores_ptr": scores_ptr,
        "query_ptr": query_ptr,
        "key_t_ptr": key_t_ptr,
        "tensor": tensor,
        "view_q": view_q,
        "view_kt": view_kt,
        "view_scores": view_scores,
        "q_mat": q_mat,
        "kt_mat": kt_mat,
        "q_tile": q_tile,
        "kt_tile": kt_tile,
        "scores_acc": scores_acc,
    }


def _softmax_meta_data(config: DenseAttentionConfig):
    dtype = pto.float16
    acc_dtype = pto.float32

    scores_ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)
    row_view = pto.SubTensorType(shape=[1, config.scores_dim], dtype=dtype)

    cfg = pto.TileConfig()
    row_tile = pto.TileType(
        shape=[1, config.scores_dim],
        valid_shape=[1, -1],
        dtype=dtype,
        memory_space="VEC",
        config=cfg,
    )
    row_tile_f32 = pto.TileType(
        shape=[1, config.scores_dim],
        valid_shape=[1, -1],
        dtype=acc_dtype,
        memory_space="VEC",
        config=cfg,
    )

    return {
        "scores_ptr": scores_ptr,
        "tensor": tensor,
        "row_view": row_view,
        "row_tile": row_tile,
        "row_tile_f32": row_tile_f32,
    }


def _pv_meta_data(*, base_m: int, base_k: int, base_n: int):
    dtype = pto.float16
    acc_dtype = pto.float32

    out_ptr = pto.PtrType(dtype)
    scores_ptr = pto.PtrType(dtype)
    value_ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)

    view_p = pto.SubTensorType(shape=[base_m, base_k], dtype=dtype)
    view_v = pto.SubTensorType(shape=[base_k, base_n], dtype=dtype)
    view_out = pto.SubTensorType(shape=[base_m, base_n], dtype=dtype)

    p_mat = pto.TileType(shape=[base_m, base_k], dtype=dtype, memory_space="MAT")
    v_mat = pto.TileType(shape=[base_k, base_n], dtype=dtype, memory_space="MAT")
    p_tile = pto.TileType(shape=[base_m, base_k], dtype=dtype, memory_space="LEFT")
    v_tile = pto.TileType(shape=[base_k, base_n], dtype=dtype, memory_space="RIGHT")
    out_acc = pto.TileType(shape=[base_m, base_n], dtype=acc_dtype, memory_space="ACC")

    return {
        "out_ptr": out_ptr,
        "scores_ptr": scores_ptr,
        "value_ptr": value_ptr,
        "tensor": tensor,
        "view_p": view_p,
        "view_v": view_v,
        "view_out": view_out,
        "p_mat": p_mat,
        "v_mat": v_mat,
        "p_tile": p_tile,
        "v_tile": v_tile,
        "out_acc": out_acc,
    }


def build_qk_stage(*, config: DenseAttentionConfig, output_dir):
    total_m_tiles = config.seq_len // config.qk_base_m
    total_n_tiles = config.scores_dim // config.qk_base_n
    total_tiles = total_m_tiles * total_n_tiles
    @jit(
        meta_data=lambda: _qk_meta_data(
            base_m=config.qk_base_m,
            base_k=config.qk_base_k,
            base_n=config.qk_base_n,
        ),
        output_dir=output_dir,
        block_dim=_launch_block_dim(total_tiles, config.qk_block_dim),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def dense_attention_qk_stage(
        scores_ptr: "scores_ptr",
        query_ptr: "query_ptr",
        key_t_ptr: "key_t_ptr",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cSeq = const(config.seq_len)
        cHead = const(config.head_dim)
        cScores = const(config.scores_dim)
        cBaseM = const(config.qk_base_m)
        cBaseN = const(config.qk_base_n)
        cBaseK = const(config.qk_base_k)
        cIter = const(config.qk_iters)
        cTotalTiles = const(total_tiles)
        cNTiles = const(total_n_tiles)

        tv_query = pto.as_tensor(tensor, ptr=query_ptr, shape=[cSeq, cHead], strides=[cHead, c1])
        tv_key_t = pto.as_tensor(tensor, ptr=key_t_ptr, shape=[cHead, cScores], strides=[cScores, c1])
        tv_scores = pto.as_tensor(tensor, ptr=scores_ptr, shape=[cSeq, cScores], strides=[cScores, c1])

        with pto.section.cube():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            q_mat_tile = pto.alloc_tile(q_mat)
            kt_mat_tile = pto.alloc_tile(kt_mat)
            q_tile_buf = pto.alloc_tile(q_tile)
            kt_tile_buf = pto.alloc_tile(kt_tile)
            scores_acc_tile = pto.alloc_tile(scores_acc)

            for logical_block in range(bid, cTotalTiles, num_blocks):
                m_idx, n_idx = _row_major_tile_indices(logical_block, cNTiles)
                m_off = m_idx * cBaseM
                n_off = n_idx * cBaseN

                sv_q0 = pto.slice_view(
                    view_q, source=tv_query, offsets=[m_off, c0], sizes=[cBaseM, cBaseK]
                )
                sv_kt0 = pto.slice_view(
                    view_kt, source=tv_key_t, offsets=[c0, n_off], sizes=[cBaseK, cBaseN]
                )
                pto.load(sv_q0, q_mat_tile)
                pto.load(sv_kt0, kt_mat_tile)
                pto.mov(q_mat_tile, q_tile_buf)
                pto.mov(kt_mat_tile, kt_tile_buf)
                pto.matmul(q_tile_buf, kt_tile_buf, scores_acc_tile)

                for i in range(c1, cIter, c1):
                    k_off = i * cBaseK
                    sv_q = pto.slice_view(
                        view_q, source=tv_query, offsets=[m_off, k_off], sizes=[cBaseM, cBaseK]
                    )
                    sv_kt = pto.slice_view(
                        view_kt, source=tv_key_t, offsets=[k_off, n_off], sizes=[cBaseK, cBaseN]
                    )

                    pto.load(sv_q, q_mat_tile)
                    pto.load(sv_kt, kt_mat_tile)
                    pto.mov(q_mat_tile, q_tile_buf)
                    pto.mov(kt_mat_tile, kt_tile_buf)
                    pto.matmul_acc(scores_acc_tile, q_tile_buf, kt_tile_buf, scores_acc_tile)

                sv_scores = pto.slice_view(
                    view_scores, source=tv_scores, offsets=[m_off, n_off], sizes=[cBaseM, cBaseN]
                )
                pto.store(scores_acc_tile, sv_scores)

    return dense_attention_qk_stage


def build_row_softmax_stage(*, config: DenseAttentionConfig, output_dir):
    @jit(
        meta_data=lambda: _softmax_meta_data(config),
        output_dir=output_dir,
        block_dim=max(1, min(config.seq_len, config.softmax_block_dim)),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def dense_attention_row_softmax(scores_ptr: "scores_ptr") -> None:
        c0 = const(0)
        c1 = const(1)
        cSeq = const(config.seq_len)
        cScores = const(config.scores_dim)
        tv_scores = pto.as_tensor(tensor, ptr=scores_ptr, shape=[cSeq, cScores], strides=[cScores, c1])

        with pto.section.vector():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            rows_per_core = pto.ceil_div(cSeq, num_blocks)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, cSeq)

            if config.softmax_fp32 or config.score_scale != 1.0:
                row_in_f16 = pto.alloc_tile(row_tile, valid_col=cScores)
                row_out_f16 = pto.alloc_tile(row_tile, valid_col=cScores)
                row_in_f32 = pto.alloc_tile(row_tile_f32, valid_col=cScores)
                row_tmp_f32 = pto.alloc_tile(row_tile_f32, valid_col=cScores)
                row_tmp2_f32 = pto.alloc_tile(row_tile_f32, valid_col=cScores)
                scalar_f32 = pto.alloc_tile(row_tile_f32, valid_col=c1)
                scalar_expanded_f32 = pto.alloc_tile(row_tile_f32, valid_col=cScores)
                for row_idx in range(row_start, row_end, c1):
                    sv_row = pto.slice_view(row_view, source=tv_scores, offsets=[row_idx, c0], sizes=[c1, cScores])
                    pto.load(sv_row, row_in_f16)
                    pto.cvt(row_in_f16, row_in_f32)
                    pto.muls(row_in_f32, const(config.score_scale, dtype=pto.float32), row_in_f32)

                    pto.row_max(row_in_f32, row_tmp_f32, scalar_f32)
                    pto.row_expand(scalar_f32, scalar_expanded_f32)
                    pto.sub(row_in_f32, scalar_expanded_f32, row_tmp_f32)
                    pto.exp(row_tmp_f32, row_tmp_f32)

                    pto.row_sum(row_tmp_f32, row_tmp2_f32, scalar_f32)
                    pto.row_expand(scalar_f32, scalar_expanded_f32)
                    pto.div(row_tmp_f32, scalar_expanded_f32, row_tmp2_f32)
                    pto.cvt(row_tmp2_f32, row_out_f16)
                    pto.store(row_out_f16, sv_row)
            else:
                row_in = pto.alloc_tile(row_tile, valid_col=cScores)
                row_tmp = pto.alloc_tile(row_tile, valid_col=cScores)
                row_tmp2 = pto.alloc_tile(row_tile, valid_col=cScores)
                scalar = pto.alloc_tile(row_tile, valid_col=c1)
                scalar_expanded = pto.alloc_tile(row_tile, valid_col=cScores)

                for row_idx in range(row_start, row_end, c1):
                    sv_row = pto.slice_view(row_view, source=tv_scores, offsets=[row_idx, c0], sizes=[c1, cScores])
                    pto.load(sv_row, row_in)

                    pto.row_max(row_in, row_tmp, scalar)
                    pto.row_expand(scalar, scalar_expanded)
                    pto.sub(row_in, scalar_expanded, row_tmp)
                    pto.exp(row_tmp, row_tmp)

                    pto.row_sum(row_tmp, row_tmp2, scalar)
                    pto.row_expand(scalar, scalar_expanded)
                    pto.div(row_tmp, scalar_expanded, row_tmp2)
                    pto.store(row_tmp2, sv_row)

    return dense_attention_row_softmax


def build_pv_stage(*, config: DenseAttentionConfig, output_dir):
    total_m_tiles = config.seq_len // config.pv_base_m
    total_n_tiles = config.head_dim // config.pv_base_n
    total_tiles = total_m_tiles * total_n_tiles
    @jit(
        meta_data=lambda: _pv_meta_data(
            base_m=config.pv_base_m,
            base_k=config.pv_base_k,
            base_n=config.pv_base_n,
        ),
        output_dir=output_dir,
        block_dim=_launch_block_dim(total_tiles, config.pv_block_dim),
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def dense_attention_pv_stage(
        out_ptr: "out_ptr",
        scores_ptr: "scores_ptr",
        value_ptr: "value_ptr",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cSeq = const(config.seq_len)
        cHead = const(config.head_dim)
        cScores = const(config.scores_dim)
        cBaseM = const(config.pv_base_m)
        cBaseN = const(config.pv_base_n)
        cBaseK = const(config.pv_base_k)
        cIter = const(config.pv_iters)
        cTotalTiles = const(total_tiles)
        cNTiles = const(total_n_tiles)

        tv_scores = pto.as_tensor(tensor, ptr=scores_ptr, shape=[cSeq, cScores], strides=[cScores, c1])
        tv_value = pto.as_tensor(tensor, ptr=value_ptr, shape=[cScores, cHead], strides=[cHead, c1])
        tv_out = pto.as_tensor(tensor, ptr=out_ptr, shape=[cSeq, cHead], strides=[cHead, c1])

        with pto.section.cube():
            bid = pto.index_cast(pto.get_block_idx())
            num_blocks = pto.index_cast(pto.get_block_num())
            p_mat_tile = pto.alloc_tile(p_mat)
            v_mat_tile = pto.alloc_tile(v_mat)
            p_tile_buf = pto.alloc_tile(p_tile)
            v_tile_buf = pto.alloc_tile(v_tile)
            out_acc_tile = pto.alloc_tile(out_acc)

            for logical_block in range(bid, cTotalTiles, num_blocks):
                m_idx, n_idx = _row_major_tile_indices(logical_block, cNTiles)
                m_off = m_idx * cBaseM
                n_off = n_idx * cBaseN

                sv_p0 = pto.slice_view(
                    view_p, source=tv_scores, offsets=[m_off, c0], sizes=[cBaseM, cBaseK]
                )
                sv_v0 = pto.slice_view(
                    view_v, source=tv_value, offsets=[c0, n_off], sizes=[cBaseK, cBaseN]
                )
                pto.load(sv_p0, p_mat_tile)
                pto.load(sv_v0, v_mat_tile)
                pto.mov(p_mat_tile, p_tile_buf)
                pto.mov(v_mat_tile, v_tile_buf)
                pto.matmul(p_tile_buf, v_tile_buf, out_acc_tile)

                for i in range(c1, cIter, c1):
                    k_off = i * cBaseK
                    sv_p = pto.slice_view(
                        view_p, source=tv_scores, offsets=[m_off, k_off], sizes=[cBaseM, cBaseK]
                    )
                    sv_v = pto.slice_view(
                        view_v, source=tv_value, offsets=[k_off, n_off], sizes=[cBaseK, cBaseN]
                    )

                    pto.load(sv_p, p_mat_tile)
                    pto.load(sv_v, v_mat_tile)
                    pto.mov(p_mat_tile, p_tile_buf)
                    pto.mov(v_mat_tile, v_tile_buf)
                    pto.matmul_acc(out_acc_tile, p_tile_buf, v_tile_buf, out_acc_tile)

                sv_out = pto.slice_view(
                    view_out, source=tv_out, offsets=[m_off, n_off], sizes=[cBaseM, cBaseN]
                )
                pto.store(out_acc_tile, sv_out)

    return dense_attention_pv_stage


class DenseAttentionPipelineWrapper:
    def __init__(self, *, config: DenseAttentionConfig, output_dir):
        config.validate()
        self._config = config
        self._output_dir = Path(output_dir)
        self._qk = build_qk_stage(config=config, output_dir=self._output_dir / "stage1_qk")
        self._softmax = build_row_softmax_stage(config=config, output_dir=self._output_dir / "stage2_softmax")
        self._pv = build_pv_stage(config=config, output_dir=self._output_dir / "stage3_pv")

    def _build(self):
        self._qk._build()
        self._softmax._build()
        self._pv._build()

    def _artifact_paths(self):
        return (*self._qk._artifact_paths(), *self._softmax._artifact_paths(), *self._pv._artifact_paths())

    @property
    def library_path(self):
        return None

    @property
    def output_dir(self):
        return str(self._output_dir)

    def __call__(self, out_ptr, scores_ptr, query_ptr, key_t_ptr, value_ptr, stream_ptr=None):
        self._qk(scores_ptr, query_ptr, key_t_ptr, stream_ptr=stream_ptr)
        self._softmax(scores_ptr, stream_ptr=stream_ptr)
        self._pv(out_ptr, scores_ptr, value_ptr, stream_ptr=stream_ptr)
