from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="moe",
    name="moe_re_routing",
    wave="wave2",
    archetype="moe-routing",
    ops_transformer_path="moe/moe_re_routing",
    blockers=[
        "ptodsl-group-routing-primitives",
        "ptoas-scf-routing-emit",
    ],
)

META["status"] = "blocked"
META["seed_variant"] = {
    "name": "fp16_moe_re_routing_with_scales",
    "shape": {
        "smoke": {"tokens": 8, "hidden": 16, "ranks": 2, "experts": 2},
        "nominal": {"tokens": 256, "hidden": 64, "ranks": 4, "experts": 4},
        "boundary": {"tokens": 128, "hidden": 128, "ranks": 4, "experts": 4},
    },
    "limits": [
        "per_token_scales is enabled for the validated slice",
        "expert_token_num_type fixed to 1 (count mode)",
        "idx_type fixed to 0 (gather indices) because Atlas A2/910B does not support idx_type=1",
        "PTO seed computes expert-major destination order directly from expert_token_num_per_rank with scalar prefix scans",
        "phase-2 rewrite follows contiguous per-core destination-row ownership",
        "nominal shape uses all requested block ids",
    ],
}
