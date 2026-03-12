from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="mc2",
    name="moe_distribute_dispatch",
    wave="wave5",
    archetype="mc2-routing",
    ops_transformer_path="mc2/moe_distribute_dispatch",
    blockers=[
        "ptodsl-multicore-collective-primitives",
        "ptodsl-routing-sort-primitives",
        "ptodsl-group-routing-primitives",
        "ptoas-memory-plan-sync-pipeline",
    ],
)

META["seed_variant"] = {
    "name": "ep_only_top1_fp16_dispatch",
    "shape": [8, 7168],
    "tiling": {
        "strategy": "destination-major local send packing with contiguous per-core token-row ownership via a flattened gather map stopgap",
        "block_dim_env": "PTO_MC2_MOE_DISPATCH_BLOCK_DIM",
    },
    "limits": [
        "world_size fixed to 8 for the first seed contract because the A2 baseline requires epWorldSize % 8 == 0",
        "EP-only path only; TP group remains empty",
        "top-1 routing only",
        "float16 only",
        "hidden size fixed to 7168 for the A2/910B baseline contract",
        "quant_mode fixed to 0",
        "local expert count fixed to 1 per rank",
        "PTO seed uses a host-precomputed destination-major flattened gather map plus host HCCL all_to_all_single",
        "validated shape currently targets 8x7168",
    ],
}
