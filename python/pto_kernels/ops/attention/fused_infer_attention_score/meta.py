from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="attention",
    name="fused_infer_attention_score",
    wave="wave3",
    archetype="attention-cache",
    ops_transformer_path="attention/fused_infer_attention_score",
    blockers=[],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "bnsd_single_block_no_quant",
    "shape": {
        "smoke": {
            "query": [1, 1, 16, 16],
            "k_cache": [1, 16, 16],
            "v_cache": [1, 16, 16],
            "block_table": [1, 1],
            "attention_out": [1, 1, 16, 16],
        },
        "nominal": {
            "query": [1, 1, 64, 64],
            "k_cache": [1, 64, 64],
            "v_cache": [1, 64, 64],
            "block_table": [1, 1],
            "attention_out": [1, 1, 64, 64],
        },
        "boundary": {
            "query": [1, 1, 32, 128],
            "k_cache": [1, 128, 128],
            "v_cache": [1, 128, 128],
            "block_table": [1, 1],
            "attention_out": [1, 1, 32, 128],
        },
    },
    "limits": [
        "constrained no-quant slice only",
        "batch=1, q_heads=kv_heads=1, total_blocks=1",
        "cache layout fixed to [blocknum, blocksize, H]",
        "input layout fixed to BNSD",
        "no mask, no rope, no shared prefix, no quant/dequant branches",
    ],
}
