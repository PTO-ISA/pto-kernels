from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="posembedding",
    name="interleave_rope",
    wave="wave1",
    archetype="rope-pos",
    ops_transformer_path="posembedding/interleave_rope",
    blockers=[
        "ptodsl-rope-generalization",
        "ptodsl-vector-queue-pipeline-surface",
        "ptodsl-vector-interleave-surface",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "bnsd_fp16_d64",
    "shape": {
        "variant_0": [1, 1, 32, 64],
        "variant_1": [2, 1, 32, 64],
    },
    "limits": [
        "BNSD layout only",
        "float16 only",
        "head_dim fixed to 64",
        "heads fixed to 1",
        "cos and sin use the same sequence length as x",
        "PTO path currently performs the interleave stage with NPU reshape+transpose preprocessing before the PTODSL rotary-half stage",
        "current PTO rewrite does not yet model the upstream vector queue and double-buffer pipeline",
    ],
}
