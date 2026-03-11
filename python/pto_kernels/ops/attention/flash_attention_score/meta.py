from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="attention",
    name="flash_attention_score",
    wave="wave3",
    archetype="attention-core",
    ops_transformer_path="attention/flash_attention_score",
    blockers=[
        "ptodsl-sparse-attention-primitives",
        "ptoas-memory-plan-sync-pipeline",
        "ptoas-a3-legality-diagnostics",
    ],
)
