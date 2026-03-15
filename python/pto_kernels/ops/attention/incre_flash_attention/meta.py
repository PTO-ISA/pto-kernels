from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="attention",
    name="incre_flash_attention",
    wave="wave3",
    archetype="attention-dense",
    ops_transformer_path="attention/incre_flash_attention",
    blockers=[],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "bnsd_no_mask_no_quant_decode",
    "shape": {
        "smoke": {
            "query": [1, 16, 1, 16],
            "key": [1, 1, 16, 16],
            "value": [1, 1, 16, 16],
            "attention_out": [1, 16, 1, 16],
        },
        "nominal": {
            "query": [1, 16, 1, 64],
            "key": [1, 1, 128, 64],
            "value": [1, 1, 128, 64],
            "attention_out": [1, 16, 1, 64],
        },
        "boundary": {
            "query": [1, 16, 1, 128],
            "key": [1, 1, 128, 128],
            "value": [1, 1, 128, 128],
            "attention_out": [1, 16, 1, 128],
        },
    },
    "limits": [
        "constrained decode-only slice with q_seq fixed to 1",
        "batch=1, q_heads=16, kv_heads=1",
        "input layout fixed to BNSD",
        "no mask, no quant, no block_table, no kv padding",
        "actual_seq_lengths fixed to the full kv sequence length",
    ],
}
