from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="mc2",
    name="moe_distribute_combine",
    wave="wave5",
    archetype="mc2-routing",
    ops_transformer_path="mc2/moe_distribute_combine",
    blockers=[
        "ptodsl-multicore-collective-primitives",
        "ptodsl-routing-sort-primitives",
        "ptodsl-group-routing-primitives",
        "ptoas-memory-plan-sync-pipeline",
    ],
)
META["status"] = "prototype"

META["seed_variant"] = {
    "name": "ep_only_top1_fp16_combine",
    "shape": [8, 7168],
    "tiling": {
        "strategy": "contiguous per-core token-row ownership over a host-precompacted reverse-route buffer with chunk-local flattened scatter indices",
        "block_dim_env": "PTO_MC2_MOE_COMBINE_BLOCK_DIM",
    },
    "limits": [
        "world_size fixed to 8 for the first seed contract because the A2 baseline requires epWorldSize % 8 == 0",
        "EP-only path only; TP group remains empty",
        "top-1 routing only",
        "float16 only",
        "hidden size fixed to 7168 for the A2/910B baseline contract",
        "PTO seed currently consumes a host-precompacted reverse-route buffer plus host-generated chunk-local flattened scatter indices instead of implementing the distributed return path",
        "validated shape currently targets 8x7168",
    ],
}
