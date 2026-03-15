from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="moe",
    name="moe_gating_top_k_softmax_v2",
    wave="wave2",
    archetype="moe-routing",
    ops_transformer_path="moe/moe_gating_top_k_softmax_v2",
    blockers=[],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "top1_fp16_2d_softmax_v2_renorm0",
    "shape": [8, 16],
    "limits": [
        "top-k fixed to 1",
        "renorm fixed to 0",
        "2D x only",
        "finishedOptional fixed to none",
        "outputSoftmaxResultFlag fixed to false",
        "validated shapes currently cover 8x16, 256x64, and 128x128",
        "renorm=1, optional softmax output, and 3D inputs remain open",
        "this host does not expose a Python-visible baseline entrypoint for v2",
    ],
}
