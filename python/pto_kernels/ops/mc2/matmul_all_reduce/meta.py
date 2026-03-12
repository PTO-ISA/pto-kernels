from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="mc2",
    name="matmul_all_reduce",
    wave="wave5",
    archetype="mc2-comm",
    ops_transformer_path="mc2/matmul_all_reduce",
    blockers=[
        "ptodsl-multicore-collective-primitives",
        "ptoas-memory-plan-sync-pipeline",
        "ptoisa-a2a3-template-gaps",
    ],
)
META["status"] = "prototype"

META["seed_variant"] = {
    "name": "dense_fp16_all_reduce",
    "shape": [128, 256, 128],
    "tiling": {
        "strategy": "ops-transformer splitM/numBlocksN turn loop for local dense matmul",
        "base_m_env": "PTO_MC2_MM_AR_BASE_M",
        "base_n_env": "PTO_MC2_MM_AR_BASE_N",
        "base_k_env": "PTO_MC2_MM_AR_BASE_K",
        "block_dim_env": "PTO_MC2_MM_AR_BLOCK_DIM",
    },
    "limits": [
        "world_size fixed to 2 for the first seed contract",
        "bias disabled",
        "x3 disabled",
        "float16 only",
        "x1 is local per-rank while x2 is replicated across ranks",
        "PTO and baseline both use host HCCL all_reduce(sum) around the local matmul path",
        "PTO local kernel mirrors the upstream splitM/numBlocksN core traversal, but not the upstream HCCL overlap pipeline",
    ],
}
