from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="attention",
    name="ring_attention_update",
    wave="wave3",
    archetype="attention-update",
    ops_transformer_path="attention/ring_attention_update",
    blockers=[
        "host-ring-attention-update-python-entrypoint",
        "ptodsl-ring-attention-update-list-surface",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "tnd_sp2_fp16_fp32_merge",
    "shape": {
        "smoke": {
            "prev_attn_out": [8, 1, 16],
            "prev_softmax_max": [8, 1, 8],
            "prev_softmax_sum": [8, 1, 8],
            "cur_attn_out": [8, 1, 16],
            "cur_softmax_max": [8, 1, 8],
            "cur_softmax_sum": [8, 1, 8],
            "attn_out": [8, 1, 16],
            "softmax_max_out": [8, 1, 8],
            "softmax_sum_out": [8, 1, 8],
        },
        "nominal": {
            "prev_attn_out": [256, 1, 64],
            "prev_softmax_max": [256, 1, 8],
            "prev_softmax_sum": [256, 1, 8],
            "cur_attn_out": [256, 1, 64],
            "cur_softmax_max": [256, 1, 8],
            "cur_softmax_sum": [256, 1, 8],
            "attn_out": [256, 1, 64],
            "softmax_max_out": [256, 1, 8],
            "softmax_sum_out": [256, 1, 8],
        },
        "boundary": {
            "prev_attn_out": [128, 1, 128],
            "prev_softmax_max": [128, 1, 8],
            "prev_softmax_sum": [128, 1, 8],
            "cur_attn_out": [128, 1, 128],
            "cur_softmax_max": [128, 1, 8],
            "cur_softmax_sum": [128, 1, 8],
            "attn_out": [128, 1, 128],
            "softmax_max_out": [128, 1, 8],
            "softmax_sum_out": [128, 1, 8],
        },
    },
    "limits": [
        "constrained TND slice with N fixed to 1",
        "softmax max/sum tensors use the validated last-dim-8 repeated contract",
        "baseline Python entrypoint is not exposed on this host",
        "PTO seed mirrors the row-wise merge formula only",
        "nominal shape uses all requested block ids on the checked-in block_dim",
    ],
}
