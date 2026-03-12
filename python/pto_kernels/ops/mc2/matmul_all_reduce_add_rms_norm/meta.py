from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="mc2",
    name="matmul_all_reduce_add_rms_norm",
    wave="wave5",
    archetype="mc2-comm-epilogue",
    ops_transformer_path="mc2/matmul_all_reduce_add_rms_norm",
    blockers=[
        "ptodsl-multicore-collective-primitives",
        "ptodsl-vector-rms-norm-generalization",
        "ptoisa-a2a3-vector-cast-io",
        "ptoas-memory-plan-sync-pipeline",
    ],
)
META["status"] = "prototype"

META["seed_variant"] = {
    "name": "dense_fp16_all_reduce_add_rms_norm",
    "shape": [128, 256, 128],
    "tiling": {
        "strategy": "ops-transformer splitM/numBlocksN local matmul plus row-wise vector add+rms_norm epilogue",
        "base_m_env": "PTO_MC2_MM_AR_BASE_M",
        "base_n_env": "PTO_MC2_MM_AR_BASE_N",
        "base_k_env": "PTO_MC2_MM_AR_BASE_K",
        "block_dim_env": "PTO_MC2_MM_AR_BLOCK_DIM",
        "epilogue_block_dim_env": "PTO_MC2_MM_ARN_BLOCK_DIM",
    },
    "limits": [
        "world_size fixed to 2 for the first seed contract",
        "bias disabled",
        "x3 disabled",
        "float16 only",
        "residual flattened to a 2D [M, N] dense tensor for the first seed",
        "gamma fixed to 1D [N]",
        "PTO local kernel mirrors the upstream splitM/numBlocksN core traversal for matmul and a separate vector epilogue for add+rms_norm",
        "HCCL all_reduce(sum) remains host-orchestrated outside the PTO kernel boundary",
    ],
}
