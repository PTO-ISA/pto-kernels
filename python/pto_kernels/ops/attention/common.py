"""Shared staged dense-attention helpers for PTO seed kernels."""

from dataclasses import dataclass
from pathlib import Path

from ptodsl import jit, pto, tile
from ptodsl import scalar as s


const = s.const


@dataclass(frozen=True)
class DenseAttentionConfig:
    seq_len: int
    head_dim: int
    scores_dim: int
    qk_base_k: int
    pv_base_k: int

    @property
    def qk_iters(self) -> int:
        return self.head_dim // self.qk_base_k

    @property
    def pv_iters(self) -> int:
        return self.scores_dim // self.pv_base_k


def _qk_meta_data(config: DenseAttentionConfig):
    dtype = pto.float16
    acc_dtype = pto.float32

    ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)

    view_q = pto.SubTensorType(shape=[config.seq_len, config.qk_base_k], dtype=dtype)
    view_kt = pto.SubTensorType(shape=[config.qk_base_k, config.scores_dim], dtype=dtype)
    view_scores = pto.SubTensorType(shape=[config.seq_len, config.scores_dim], dtype=dtype)

    q_mat = pto.TileBufType(shape=[config.seq_len, config.qk_base_k], dtype=dtype, memory_space="MAT")
    kt_mat = pto.TileBufType(shape=[config.qk_base_k, config.scores_dim], dtype=dtype, memory_space="MAT")
    q_tile = pto.TileBufType(shape=[config.seq_len, config.qk_base_k], dtype=dtype, memory_space="LEFT")
    kt_tile = pto.TileBufType(shape=[config.qk_base_k, config.scores_dim], dtype=dtype, memory_space="RIGHT")
    scores_acc = pto.TileBufType(
        shape=[config.seq_len, config.scores_dim], dtype=acc_dtype, memory_space="ACC"
    )

    return {
        "ptr": ptr,
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

    ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)
    row_view = pto.SubTensorType(shape=[1, config.scores_dim], dtype=dtype)

    cfg = pto.TileBufConfig()
    row_tile = pto.TileBufType(
        shape=[1, config.scores_dim],
        valid_shape=[1, -1],
        dtype=dtype,
        memory_space="VEC",
        config=cfg,
    )

    return {
        "ptr": ptr,
        "tensor": tensor,
        "row_view": row_view,
        "row_tile": row_tile,
    }


def _pv_meta_data(config: DenseAttentionConfig):
    dtype = pto.float16
    acc_dtype = pto.float32

    ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)

    view_p = pto.SubTensorType(shape=[config.seq_len, config.pv_base_k], dtype=dtype)
    view_v = pto.SubTensorType(shape=[config.pv_base_k, config.head_dim], dtype=dtype)
    view_out = pto.SubTensorType(shape=[config.seq_len, config.head_dim], dtype=dtype)

    p_mat = pto.TileBufType(shape=[config.seq_len, config.pv_base_k], dtype=dtype, memory_space="MAT")
    v_mat = pto.TileBufType(shape=[config.pv_base_k, config.head_dim], dtype=dtype, memory_space="MAT")
    p_tile = pto.TileBufType(shape=[config.seq_len, config.pv_base_k], dtype=dtype, memory_space="LEFT")
    v_tile = pto.TileBufType(shape=[config.pv_base_k, config.head_dim], dtype=dtype, memory_space="RIGHT")
    out_acc = pto.TileBufType(shape=[config.seq_len, config.head_dim], dtype=acc_dtype, memory_space="ACC")

    return {
        "ptr": ptr,
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
    @jit(
        meta_data=lambda: _qk_meta_data(config),
        output_dir=output_dir,
        block_dim=1,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def dense_attention_qk_stage(scores_ptr: "ptr", query_ptr: "ptr", key_t_ptr: "ptr") -> None:
        c0 = const(0)
        c1 = const(1)
        cSeq = const(config.seq_len)
        cHead = const(config.head_dim)
        cScores = const(config.scores_dim)
        cBaseK = const(config.qk_base_k)
        cIter = const(config.qk_iters)

        tv_query = pto.as_tensor(tensor, ptr=query_ptr, shape=[cSeq, cHead], strides=[cHead, c1])
        tv_key_t = pto.as_tensor(tensor, ptr=key_t_ptr, shape=[cHead, cScores], strides=[cScores, c1])
        tv_scores = pto.as_tensor(tensor, ptr=scores_ptr, shape=[cSeq, cScores], strides=[cScores, c1])

        with pto.cube_section():
            q_mat_tile = pto.alloc_tile(q_mat)
            kt_mat_tile = pto.alloc_tile(kt_mat)
            q_tile_buf = pto.alloc_tile(q_tile)
            kt_tile_buf = pto.alloc_tile(kt_tile)
            scores_acc_tile = pto.alloc_tile(scores_acc)

            for i in pto.range(c0, cIter, c1):
                k_off = i * cBaseK
                sv_q = pto.slice_view(view_q, source=tv_query, offsets=[c0, k_off], sizes=[cSeq, cBaseK])
                sv_kt = pto.slice_view(view_kt, source=tv_key_t, offsets=[k_off, c0], sizes=[cBaseK, cScores])

                pto.load(sv_q, q_mat_tile)
                pto.load(sv_kt, kt_mat_tile)
                tile.mov(q_mat_tile, q_tile_buf)
                tile.mov(kt_mat_tile, kt_tile_buf)

                pto.cond(
                    s.eq(i, c0),
                    lambda: tile.matmul(q_tile_buf, kt_tile_buf, scores_acc_tile),
                    lambda: tile.matmul_acc(scores_acc_tile, q_tile_buf, kt_tile_buf, scores_acc_tile),
                )

            sv_scores = pto.slice_view(
                view_scores, source=tv_scores, offsets=[c0, c0], sizes=[cSeq, cScores]
            )
            pto.store(scores_acc_tile, sv_scores)

    return dense_attention_qk_stage


def build_row_softmax_stage(*, config: DenseAttentionConfig, output_dir):
    @jit(
        meta_data=lambda: _softmax_meta_data(config),
        output_dir=output_dir,
        block_dim=1,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def dense_attention_row_softmax(scores_ptr: "ptr") -> None:
        c0 = const(0)
        c1 = const(1)
        cSeq = const(config.seq_len)
        cScores = const(config.scores_dim)

        tv_scores = pto.as_tensor(tensor, ptr=scores_ptr, shape=[cSeq, cScores], strides=[cScores, c1])

        with pto.vector_section():
            row_in = pto.alloc_tile(row_tile, valid_col=cScores)
            row_tmp = pto.alloc_tile(row_tile, valid_col=cScores)
            row_tmp2 = pto.alloc_tile(row_tile, valid_col=cScores)
            scalar = pto.alloc_tile(row_tile, valid_col=c1)
            scalar_expanded = pto.alloc_tile(row_tile, valid_col=cScores)

            for row_idx in pto.range(c0, cSeq, c1):
                sv_row = pto.slice_view(row_view, source=tv_scores, offsets=[row_idx, c0], sizes=[c1, cScores])
                pto.load(sv_row, row_in)

                tile.row_max(row_in, row_tmp, scalar)
                tile.row_expand(scalar, scalar_expanded)
                tile.sub(row_in, scalar_expanded, row_tmp)
                tile.exp(row_tmp, row_tmp)

                tile.row_sum(row_tmp, row_tmp2, scalar)
                tile.row_expand(scalar, scalar_expanded)
                tile.div(row_tmp, scalar_expanded, row_tmp2)
                pto.store(row_tmp2, sv_row)

    return dense_attention_row_softmax


def build_pv_stage(*, config: DenseAttentionConfig, output_dir):
    @jit(
        meta_data=lambda: _pv_meta_data(config),
        output_dir=output_dir,
        block_dim=1,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    )
    def dense_attention_pv_stage(out_ptr: "ptr", scores_ptr: "ptr", value_ptr: "ptr") -> None:
        c0 = const(0)
        c1 = const(1)
        cSeq = const(config.seq_len)
        cHead = const(config.head_dim)
        cBaseK = const(config.pv_base_k)
        cIter = const(config.pv_iters)

        tv_scores = pto.as_tensor(
            tensor, ptr=scores_ptr, shape=[cSeq, const(config.scores_dim)], strides=[const(config.scores_dim), c1]
        )
        tv_value = pto.as_tensor(
            tensor, ptr=value_ptr, shape=[const(config.scores_dim), cHead], strides=[cHead, c1]
        )
        tv_out = pto.as_tensor(tensor, ptr=out_ptr, shape=[cSeq, cHead], strides=[cHead, c1])

        with pto.cube_section():
            p_mat_tile = pto.alloc_tile(p_mat)
            v_mat_tile = pto.alloc_tile(v_mat)
            p_tile_buf = pto.alloc_tile(p_tile)
            v_tile_buf = pto.alloc_tile(v_tile)
            out_acc_tile = pto.alloc_tile(out_acc)

            for i in pto.range(c0, cIter, c1):
                k_off = i * cBaseK
                sv_p = pto.slice_view(view_p, source=tv_scores, offsets=[c0, k_off], sizes=[cSeq, cBaseK])
                sv_v = pto.slice_view(view_v, source=tv_value, offsets=[k_off, c0], sizes=[cBaseK, cHead])

                pto.load(sv_p, p_mat_tile)
                pto.load(sv_v, v_mat_tile)
                tile.mov(p_mat_tile, p_tile_buf)
                tile.mov(v_mat_tile, v_tile_buf)

                pto.cond(
                    s.eq(i, c0),
                    lambda: tile.matmul(p_tile_buf, v_tile_buf, out_acc_tile),
                    lambda: tile.matmul_acc(out_acc_tile, p_tile_buf, v_tile_buf, out_acc_tile),
                )

            sv_out = pto.slice_view(view_out, source=tv_out, offsets=[c0, c0], sizes=[cSeq, cHead])
            pto.store(out_acc_tile, sv_out)

    return dense_attention_pv_stage


class DenseAttentionPipelineWrapper:
    def __init__(self, *, config: DenseAttentionConfig, output_dir):
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
