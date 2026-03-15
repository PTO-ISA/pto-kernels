from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="attention",
    name="prompt_flash_attention",
    wave="wave3",
    archetype="attention-dense",
    ops_transformer_path="attention/prompt_flash_attention",
    blockers=[],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "bnsd_no_mask_no_quant",
    "shape": {
        "smoke": {
            "query": [1, 1, 16, 16],
            "key": [1, 1, 16, 16],
            "value": [1, 1, 16, 16],
            "attention_out": [1, 1, 16, 16],
        },
        "nominal": {
            "query": [1, 1, 64, 64],
            "key": [1, 1, 128, 64],
            "value": [1, 1, 128, 64],
            "attention_out": [1, 1, 64, 64],
        },
        "boundary": {
            "query": [1, 1, 32, 128],
            "key": [1, 1, 128, 128],
            "value": [1, 1, 128, 128],
            "attention_out": [1, 1, 32, 128],
        },
    },
    "limits": [
        "constrained no-mask no-quant slice only",
        "batch=1, q_heads=kv_heads=1",
        "input layout fixed to BNSD",
        "actual_seq_lengths and actual_seq_lengths_kv fixed to full lengths",
        "no pse, no mask, no sparse optimization, no page attention",
    ],
}
