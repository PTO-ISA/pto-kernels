from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="posembedding",
    name="apply_rotary_pos_emb",
    wave="wave1",
    archetype="rope-pos",
    ops_transformer_path="posembedding/apply_rotary_pos_emb",
    blockers=[
        "ptodsl-rope-generalization",
        "ptodsl-vector-queue-pipeline-surface",
    ],
)

META["seed_variant"] = {
    "name": "tnd_bsnd_half_fp16",
    "shape": {
        "TND": [64, 1, 128],
        "BSND": [2, 32, 1, 128],
    },
    "limits": [
        "layouts covered: TND and BSND",
        "rotary_mode fixed to half",
        "float16 only",
        "query_heads and key_heads fixed to 1",
        "head_dim fixed to 128",
        "current PTO rewrite matches upstream contiguous per-core row chunk ownership for the validated 64-row seed",
        "current PTO rewrite does not yet model the upstream double-buffer queue pipeline for vector copy/compute overlap",
    ],
}
