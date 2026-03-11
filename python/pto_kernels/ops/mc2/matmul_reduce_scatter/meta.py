from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="mc2",
    name="matmul_reduce_scatter",
    wave="wave5",
    archetype="mc2-comm",
    ops_transformer_path="mc2/matmul_reduce_scatter",
    blockers=[
        "ptodsl-multicore-collective-primitives",
        "ptoas-memory-plan-sync-pipeline",
        "ptoisa-a2a3-template-gaps",
    ],
)
