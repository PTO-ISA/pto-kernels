from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="moe",
    name="moe_gating_top_k",
    wave="wave2",
    archetype="moe-routing",
    ops_transformer_path="moe/moe_gating_top_k",
    blockers=[],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "top1_sigmoid_no_group_no_bias",
    "shape": [8, 16],
    "limits": [
        "top-k fixed to 1",
        "groupCount fixed to 1",
        "kGroup fixed to 1",
        "groupSelectMode fixed to 0",
        "biasOptional fixed to None",
        "normType fixed to sigmoid",
        "renorm fixed to 0",
        "outFlag fixed to false",
        "routedScalingFactor fixed to 1.0",
        "validated shapes currently cover 8x16, 256x64, and 128x128",
    ],
}
