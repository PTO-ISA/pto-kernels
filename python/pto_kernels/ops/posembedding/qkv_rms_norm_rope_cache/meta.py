from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="posembedding",
    name="qkv_rms_norm_rope_cache",
    wave="wave1",
    archetype="rope-kvcache",
    ops_transformer_path="posembedding/qkv_rms_norm_rope_cache",
    blockers=[
        "ops-transformer-qkv-rms-norm-rope-cache-python-entrypoint-gap",
        "ptodsl-rope-generalization",
        "ptodsl-vector-queue-pipeline-surface",
        "ptodsl-cache-update-primitives",
        "ptodsl-vector-rms-norm-generalization",
        "ptoisa-a2a3-vec-quant-store",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "bsnd_2d_fp16_qkv_rms_norm_rope_cache_d64",
    "shape": {
        "variant_0": {
            "qkv": [2, 192],
            "q_gamma": [64],
            "k_gamma": [64],
            "cos": [2, 64],
            "sin": [2, 64],
            "k_cache": [2, 8, 1, 64],
            "v_cache": [2, 8, 1, 64],
        },
        "variant_1": {
            "qkv": [4, 192],
            "q_gamma": [64],
            "k_gamma": [64],
            "cos": [4, 64],
            "sin": [4, 64],
            "k_cache": [4, 8, 1, 64],
            "v_cache": [4, 8, 1, 64],
        },
    },
    "limits": [
        "bounded PTO-vs-reference slice on 910B",
        "2D token-major slice only",
        "float16 input only",
        "single q-head and single k/v-head only",
        "size_splits fixed to [64, 64, 64]",
        "half-and-half rope only",
        "unit quant scales only",
        "cache_mode fixed to contiguous",
        "baseline runtime entrypoint is unavailable on this host",
        "current PTO staged slice is correctness-green on the validated shapes",
        "validated only for indices[row] == row in the current constrained seed",
    ],
}
