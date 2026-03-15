from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="moe",
    name="moe_finalize_routing_v2",
    wave="wave2",
    archetype="moe-routing",
    ops_transformer_path="moe/moe_finalize_routing_v2",
    blockers=[],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "top1_fp16_finalize_routing_v2",
    "shape": {
        "smoke": {"tokens": 16, "hidden": 16, "experts": 4},
        "nominal": {"tokens": 256, "hidden": 64, "experts": 4},
        "boundary": {"tokens": 128, "hidden": 128, "experts": 4},
    },
    "limits": [
        "top-k fixed to 1",
        "x1Optional and x2Optional both required in the validated slice",
        "biasOptional and scalesOptional both required in the validated slice",
        "dropPadMode fixed to 0",
        "PTO seed currently consumes direct scalar routing indices for expanded rows and expert bias rows",
        "phase-2 rewrite follows contiguous per-core token-row ownership on the combine stage",
        "nominal shape uses all requested block ids",
        "this host does not expose a Python-visible baseline entrypoint for v2",
    ],
}
