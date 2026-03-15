from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="moe",
    name="moe_token_permute_with_routing_map",
    wave="wave2",
    archetype="moe-routing",
    ops_transformer_path="moe/moe_token_permute_with_routing_map",
    blockers=[
        "ptodsl-routing-sort-primitives",
        "ptodsl-group-routing-primitives",
    ],
)

META["seed_variant"] = {
    "name": "top1_fp16_routing_map",
    "shape": [8, 16],
    "limits": [
        "top-1 routing map only",
        "probsOptional fixed to none",
        "dropAndPad fixed to false",
        "routing_map fixed to int8 with values {0,1}",
        "numOutTokens fixed to tokens_num",
        "PTO seed currently consumes host-precomputed gather and inverse-order maps for routing-map reorder",
        "validated shapes currently cover 8x16x4, 256x64x8, and 128x128x8",
    ],
}
