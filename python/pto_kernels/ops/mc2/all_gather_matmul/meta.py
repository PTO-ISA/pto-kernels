from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="mc2",
    name="all_gather_matmul",
    wave="wave5",
    archetype="mc2-comm",
    ops_transformer_path="mc2/all_gather_matmul",
    blockers=[
        "ptodsl-multicore-collective-primitives",
        "ptoas-memory-plan-sync-pipeline",
        "ptoisa-a2a3-template-gaps",
    ],
)

META["seed_variant"] = {
    "name": "dense_fp16_all_gather_matmul",
    "shape": [128, 256, 128],
    "limits": [
        "world_size fixed to 2 for the first seed contract",
        "only gather_index=0 and gather_output=true",
        "bias disabled",
        "float16 only",
        "x1 and x2 fixed to 2D dense ND tensors",
        "PTO seed uses a host HCCL all_gather and computes the gathered global matmul in PTO",
        "phase-2 rewrite follows the upstream local-first wrapped rank-chunk traversal across the validated 128x256x128 and 256x256x128 shapes",
    ],
}
