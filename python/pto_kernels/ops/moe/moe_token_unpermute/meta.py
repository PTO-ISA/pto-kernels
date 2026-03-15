from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="moe",
    name="moe_token_unpermute",
    wave="wave2",
    archetype="moe-routing",
    ops_transformer_path="moe/moe_token_unpermute",
    blockers=[
        "ptodsl-routing-sort-primitives",
        "ptodsl-group-routing-primitives",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "top1_fp16_inverse_permute",
    "shape": {
        "smoke": [8, 16],
        "nominal": [256, 64],
        "boundary": [128, 128],
    },
    "limits": [
        "top-1 routing only",
        "probsOptional fixed to None",
        "paddedMode fixed to false",
        "restoreShapeOptional fixed to None",
        "indices fixed to a permutation of [0, tokens)",
        "PTO seed currently consumes a host-precomputed inverse gather map for token restore",
        "phase-2 rewrite follows contiguous per-core token-row ownership on the restore stage",
        "nominal shape uses all requested block ids",
    ],
}
