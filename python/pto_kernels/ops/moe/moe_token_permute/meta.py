from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="moe",
    name="moe_token_permute",
    wave="wave2",
    archetype="moe-routing",
    ops_transformer_path="moe/moe_token_permute",
    blockers=[
        "ptodsl-routing-sort-primitives",
        "ptodsl-group-routing-primitives",
    ],
)

META["seed_variant"] = {
    "name": "top1_fp16_argsort",
    "shape": [8, 16],
    "limits": [
        "top-1 routing only",
        "1D int32 indices only",
        "indices fixed to a permutation of [0, tokens)",
        "padded_mode fixed to false",
        "num_out_tokens fixed to 0",
        "PTO seed currently consumes a host-precomputed gather map for token reorder",
        "phase-2 rewrite now follows ops-transformer copy ownership more closely with contiguous per-core token-row ownership on the permute stage",
        "validated shapes currently cover 8x16, 16x16, and 16x32",
    ],
}
