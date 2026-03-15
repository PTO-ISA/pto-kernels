from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="moe",
    name="moe_token_permute_with_routing_map_grad",
    wave="wave2",
    archetype="moe-routing",
    ops_transformer_path="moe/moe_token_permute_with_routing_map_grad",
    blockers=[],
)

META["seed_variant"] = {
    "name": "top1_fp16_routing_map_grad",
    "shape": [8, 16],
    "limits": [
        "top-1 routing map only",
        "probsOptional fixed to none",
        "dropAndPad fixed to false",
        "routing_map fixed to int8 with values {0,1}",
        "sortedIndices input follows the Python-visible top-1 host contract",
        "validated shapes currently cover 8x16x4, 256x64x8, and 128x128x8",
    ],
}
META["status"] = "prototype"
