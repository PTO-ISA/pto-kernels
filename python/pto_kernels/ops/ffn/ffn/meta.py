from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="ffn",
    name="ffn",
    wave="wave1",
    archetype="ffn",
    ops_transformer_path="ffn/ffn",
    blockers=[
        "ptoas-memory-plan-sync-pipeline",
        "ptoas-a3-legality-diagnostics",
    ],
)
