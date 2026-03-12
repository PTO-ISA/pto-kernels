from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="attention",
    name="flash_attention_score",
    wave="wave3",
    archetype="attention-core",
    ops_transformer_path="attention/flash_attention_score",
    blockers=[
        "ptodsl-sparse-attention-primitives",
        "ptodsl-attention-softmax-primitives",
        "ptoas-memory-plan-sync-pipeline",
    ],
)

META["seed_variant"] = {
    "name": "dense_bnsd_fp16",
    "shape": [1, 1, 32, 64],
    "limits": [
        "input_layout fixed to BNSD",
        "no masks, prefix, rope, or dropout",
        "sparse_mode fixed to 0",
        "float16 only",
        "dense self-attention only",
        "PTO seed runs as a staged QK matmul, row-wise softmax, and PV matmul path",
        "PTO benchmark pre-scales the query input by 1 / sqrt(64) to match the attention contract",
    ],
}
