from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="moe",
    name="moe_token_permute",
    wave="wave2",
    archetype="moe-routing",
    ops_transformer_path="moe/moe_token_permute",
    blockers=[
        "ptoas-memory-plan-sync-pipeline",
    ],
)
