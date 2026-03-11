from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="posembedding",
    name="apply_rotary_pos_emb",
    wave="wave1",
    archetype="rope-pos",
    ops_transformer_path="posembedding/apply_rotary_pos_emb",
    blockers=[
        "ptodsl-rope-layout-primitives",
        "ptoas-a3-legality-diagnostics",
        "ptoisa-a2a3-template-gaps",
    ],
)
