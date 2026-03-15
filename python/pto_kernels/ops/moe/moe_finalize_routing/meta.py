from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="moe",
    name="moe_finalize_routing",
    wave="wave2",
    archetype="moe-routing",
    ops_transformer_path="moe/moe_finalize_routing",
    blockers=[
        "ptodsl-routing-sort-primitives",
        "ptodsl-group-routing-primitives",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "top1_fp16_finalize_routing",
    "shape": {
        "smoke": {"tokens": 16, "hidden": 16, "experts": 4},
        "nominal": {"tokens": 256, "hidden": 64, "experts": 4},
        "boundary": {"tokens": 128, "hidden": 128, "experts": 4},
    },
    "limits": [
        "top-1 routing only",
        "x2Optional fixed to None",
        "dropPadMode fixed to 0",
        "PTO seed currently consumes direct scalar routing indices for expanded rows and expert bias rows",
        "phase-2 rewrite follows contiguous per-core token-row ownership on the combine stage",
        "nominal shape uses all requested block ids",
    ],
}
