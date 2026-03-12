from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="mc2",
    name="inplace_matmul_all_reduce_add_rms_norm",
    wave="wave5",
    archetype="mc2-comm-epilogue-inplace",
    ops_transformer_path="mc2/inplace_matmul_all_reduce_add_rms_norm",
    blockers=[
        "ptodsl-multicore-collective-primitives",
        "ptodsl-inplace-buffer-alias-contract",
        "ptoas-memory-plan-sync-pipeline",
    ],
)
META["status"] = "prototype"

META["seed_variant"] = {
    "name": "dense_fp16_inplace_all_reduce_add_rms_norm",
    "shape": [128, 256, 128],
    "tiling": {
        "strategy": "ops-transformer splitM/numBlocksN local matmul plus row-wise vector inplace add+rms_norm epilogue",
        "base_m_env": "PTO_MC2_MM_AR_BASE_M",
        "base_n_env": "PTO_MC2_MM_AR_BASE_N",
        "base_k_env": "PTO_MC2_MM_AR_BASE_K",
        "block_dim_env": "PTO_MC2_MM_AR_BLOCK_DIM",
        "epilogue_block_dim_env": "PTO_MC2_IMM_ARN_BLOCK_DIM",
    },
    "limits": [
        "world_size fixed to 2 for the first seed contract",
        "bias disabled",
        "x3 disabled",
        "float16 only",
        "residual flattened to a 2D [M, N] dense tensor for the first seed",
        "gamma fixed to 1D [N]",
        "PTO local kernel mirrors the upstream splitM/numBlocksN core traversal for matmul and uses a separate vector epilogue output plus host copy-back to model the inplace residual contract safely on the current seed",
        "HCCL all_reduce(sum) remains host-orchestrated outside the PTO kernel boundary",
    ],
}
