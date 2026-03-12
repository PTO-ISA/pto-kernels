from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="mc2",
    name="grouped_mat_mul_all_reduce",
    wave="wave5",
    archetype="mc2-comm",
    ops_transformer_path="mc2/grouped_mat_mul_all_reduce",
    blockers=[
        "ptodsl-multicore-collective-primitives",
        "ptodsl-group-routing-primitives",
        "ptoas-memory-plan-sync-pipeline",
    ],
)

META["seed_variant"] = {
    "name": "single_group_fp16_all_reduce",
    "shape": [256, 256, 128],
    "tiling": {
        "strategy": "ops-transformer splitM/numBlocksN turn loop for one dense group",
        "base_m_env": "PTO_MC2_GMM_AR_BASE_M",
        "base_n_env": "PTO_MC2_GMM_AR_BASE_N",
        "base_k_env": "PTO_MC2_GMM_AR_BASE_K",
        "block_dim_env": "PTO_MC2_GMM_AR_BLOCK_DIM",
    },
    "limits": [
        "world_size fixed to 2 for the first seed contract",
        "single dense group only",
        "split_item equivalent to 0",
        "bias disabled",
        "float16 only",
        "PTO and baseline both use host HCCL all_reduce(sum) around the local grouped-matmul path",
        "PTO local kernel mirrors the upstream turn-based splitM/numBlocksN core traversal, but not the upstream HCCL overlap pipeline",
    ],
}
