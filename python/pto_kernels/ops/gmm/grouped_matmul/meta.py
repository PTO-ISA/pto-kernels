from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="gmm",
    name="grouped_matmul",
    wave="wave1",
    archetype="gmm",
    ops_transformer_path="gmm/grouped_matmul",
    blockers=[
        "ptodsl-group-routing-primitives",
        "ptodsl-cube-preload-pipeline-surface",
    ],
)
META["status"] = "prototype"

META["seed_variant"] = {
    "name": "dense_single_weight_bf16_to_bf16",
    "shape": [128, 128, 128],
    "tiling": {
        "strategy": "ops-transformer basic-block split with diagonal block traversal",
        "base_m_env": "PTO_GROUPED_MATMUL_BASE_M",
        "base_n_env": "PTO_GROUPED_MATMUL_BASE_N",
        "base_k_env": "PTO_GROUPED_MATMUL_BASE_K",
        "block_dim_env": "PTO_GROUPED_MATMUL_BLOCK_DIM",
    },
    "limits": [
        "single batch only",
        "single dense weight only",
        "baseline path uses a single 3D weight tensor to satisfy aclnnGroupedMatmulV5",
        "no bias/quantization/activation",
        "pto path stores ACC output directly to bf16 GM for parity with the baseline contract",
        "pto path mirrors ops-transformer basic-block scheduling but does not yet model the upstream async preload callback pipeline",
    ],
}
