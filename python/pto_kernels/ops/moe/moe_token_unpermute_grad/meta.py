from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="moe",
    name="moe_token_unpermute_grad",
    wave="wave2",
    archetype="moe-routing-grad",
    ops_transformer_path="moe/moe_token_unpermute_grad",
    blockers=[
        "ptodsl-routing-sort-primitives",
        "ptodsl-group-routing-primitives",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "top1_fp16_unpermute_grad_no_probs",
    "shape": {
        "smoke": {"tokens": 8, "hidden": 16},
        "nominal": {"tokens": 256, "hidden": 64},
        "boundary": {"tokens": 128, "hidden": 128},
    },
    "limits": [
        "top-1 routing only",
        "probsOptional fixed to None",
        "paddedMode fixed to false",
        "restoreShapeOptional fixed to None",
        "PTO seed uses direct scalar routing indices and row-at-a-time GM views",
        "nominal shape uses all requested block ids",
    ],
}
