from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="moe",
    name="moe_init_routing_v2",
    wave="wave2",
    archetype="moe-routing",
    ops_transformer_path="moe/moe_init_routing_v2",
    blockers=[
        "ptodsl-routing-sort-primitives",
        "ptodsl-group-routing-primitives",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "top1_fp16_init_routing_v2_grouped_expert_idx",
    "shape": {
        "smoke": {"tokens": 16, "hidden": 16, "experts": 4},
        "nominal": {"tokens": 256, "hidden": 64, "experts": 8},
        "boundary": {"tokens": 128, "hidden": 128, "experts": 8},
    },
    "limits": [
        "top-1 routing only",
        "dropless mode only",
        "expert_idx is pre-grouped by expert on input",
        "PTO seed covers copy plus expert count/cumsum and does not sort on-device yet",
        "nominal shape uses all requested block ids for the expanded_x copy path",
    ],
}
