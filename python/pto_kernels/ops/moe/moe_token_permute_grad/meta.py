from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="moe",
    name="moe_token_permute_grad",
    wave="wave2",
    archetype="moe-routing",
    ops_transformer_path="moe/moe_token_permute_grad",
    blockers=[
        "ptodsl-routing-sort-primitives",
        "ptodsl-group-routing-primitives",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "top1_fp16_permute_grad_no_probs",
    "shape": {
        "smoke": {"tokens": 8, "hidden": 16, "topk": 1},
        "nominal": {"tokens": 256, "hidden": 64, "topk": 1},
        "boundary": {"tokens": 128, "hidden": 128, "topk": 1},
    },
    "limits": [
        "probsOptional fixed to None",
        "paddedMode fixed to false",
        "current Python-visible 910B host contract behaves as a top-1 gather path",
        "indices input is ignored by the validated no-probs host contract on this machine",
        "PTO seed computes direct row gather from sortedIndices in PTODSL",
        "phase-2 rewrite follows contiguous per-core token-row ownership on the gradient reduce stage",
        "nominal shape uses all requested block ids",
    ],
}
