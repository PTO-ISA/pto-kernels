from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="moe",
    name="moe_token_unpermute_with_routing_map_grad",
    wave="wave2",
    archetype="moe-routing",
    ops_transformer_path="moe/moe_token_unpermute_with_routing_map_grad",
    blockers=[
        "ptodsl-routing-sort-primitives",
        "ptodsl-group-routing-primitives",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "top1_fp16_routing_map_grad_no_probs",
    "shape": [8, 16],
    "limits": [
        "top-1 routing map only",
        "probsOptional fixed to none",
        "dropAndPad fixed to false",
        "routing_map fixed to int8 with values {0,1}",
        "restoreShape fixed to [tokens_num, hidden_size]",
        "PTO seed currently consumes a host-precomputed out_index row map for routing-map unpermute grad",
        "validated shapes currently cover 8x16x4, 256x64x8, and 128x128x8",
        "probs_grad is treated as unsupported in the no-probs host contract",
    ],
}
