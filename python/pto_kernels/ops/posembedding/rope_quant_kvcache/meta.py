from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="posembedding",
    name="rope_quant_kvcache",
    wave="wave1",
    archetype="rope-kvcache",
    ops_transformer_path="posembedding/rope_quant_kvcache",
    blockers=[
        "ptodsl-rope-generalization",
        "ptodsl-vector-queue-pipeline-surface",
        "ptodsl-cache-update-primitives",
        "ptoisa-a2a3-vec-quant-store",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "bsnd_2d_fp16_unit_scale_d64",
    "shape": {
        "variant_0": {"x": [2, 192], "cos": [2, 64], "sin": [2, 64], "k_cache": [2, 8, 1, 64], "v_cache": [2, 8, 1, 64]},
        "variant_1": {"x": [4, 192], "cos": [4, 64], "sin": [4, 64], "k_cache": [4, 8, 1, 64], "v_cache": [4, 8, 1, 64]},
    },
    "limits": [
        "2D token-major slice only",
        "float16 input only",
        "size_splits fixed to [64, 64, 64]",
        "single q-head and single k/v-head only",
        "static unit quant scales only",
        "cache_mode fixed to contiguous",
        "current staged PTO path keeps split/rope in a rotary kernel and explicit fp16->int8 tcvt plus contiguous cache writeback in a cache kernel",
        "A2/A3 does not currently expose a fused vec quantized store path through PTO-ISA/PTOAS; cache writeback therefore uses legal tcvt + tstore",
        "validated only for indices[row] == row in the current constrained seed",
    ],
}
