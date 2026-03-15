from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="moe",
    name="moe_gating_top_k_softmax",
    wave="wave2",
    archetype="moe-routing",
    ops_transformer_path="moe/moe_gating_top_k_softmax",
    blockers=[],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "top1_fp16_2d_softmax",
    "shape": [8, 16],
    "limits": [
        "top-k fixed to 1",
        "2D x only",
        "finishedOptional fixed to none",
        "rowIdxOut uses the Python-visible no-finished contract",
        "validated shapes currently cover 8x16, 256x64, and 128x128",
        "generalized top-k and finished masking remain open",
    ],
}
