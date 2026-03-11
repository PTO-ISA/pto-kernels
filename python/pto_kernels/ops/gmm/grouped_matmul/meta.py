from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="gmm",
    name="grouped_matmul",
    wave="wave1",
    archetype="gmm",
    ops_transformer_path="gmm/grouped_matmul",
    blockers=[
        "ptoas-memory-plan-sync-pipeline",
        "ptoisa-a2a3-template-gaps",
    ],
)
