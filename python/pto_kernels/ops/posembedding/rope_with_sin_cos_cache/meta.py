from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="posembedding",
    name="rope_with_sin_cos_cache",
    wave="wave1",
    archetype="rope-cache-read",
    ops_transformer_path="posembedding/rope_with_sin_cos_cache",
    blockers=[
        "ptodsl-rope-generalization",
        "ptodsl-vector-queue-pipeline-surface",
        "ptodsl-cache-update-primitives",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "nd_fp16_half_rope_cache_read_d64",
    "shape": {
        "variant_0": {
            "positions": [2],
            "query": [2, 64],
            "key": [2, 64],
            "cos_sin_cache": [8, 128],
            "query_out": [2, 64],
            "key_out": [2, 64],
        },
        "variant_1": {
            "positions": [4],
            "query": [4, 64],
            "key": [4, 64],
            "cos_sin_cache": [8, 128],
            "query_out": [4, 64],
            "key_out": [4, 64],
        },
    },
    "limits": [
        "2D ND token-major slice only",
        "float16 query/key and cache only",
        "head_size fixed to 64",
        "half-rope (NeoX) mode only",
        "mrope disabled",
        "current constrained PTO seed requires positions[row] == row",
        "cosSinCache rows are consumed directly in PTO source; dynamic cache-index generalization remains open",
    ],
}
