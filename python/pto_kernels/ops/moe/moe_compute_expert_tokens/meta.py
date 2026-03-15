from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="moe",
    name="moe_compute_expert_tokens",
    wave="wave2",
    archetype="moe-routing",
    ops_transformer_path="moe/moe_compute_expert_tokens",
    blockers=[
        "ptodsl-group-routing-primitives",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "sorted_expert_upper_bound_int32",
    "shape": {
        "smoke": {"rows": 64, "num_experts": 8},
        "nominal": {"rows": 4096, "num_experts": 64},
        "boundary": {"rows": 8192, "num_experts": 128},
    },
    "limits": [
        "sortedExperts must already be sorted ascending",
        "dtype fixed to int32",
        "PTO seed computes direct upper-bound positions per expert in PTODSL",
        "phase-2 rewrite follows contiguous per-core expert-range ownership",
        "nominal shape uses all requested block ids",
    ],
}
