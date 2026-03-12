from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="mc2",
    name="matmul_reduce_scatter",
    wave="wave5",
    archetype="mc2-comm",
    ops_transformer_path="mc2/matmul_reduce_scatter",
    blockers=[
        "ptodsl-multicore-collective-primitives",
        "ptodsl-mc2-distributed-launch-contract",
        "ptoas-memory-plan-sync-pipeline",
        "ptoisa-a2a3-template-gaps",
    ],
)

META["seed_variant"] = {
    "name": "dense_fp16_reduce_scatter",
    "shape": [128, 256, 128],
    "limits": [
        "world_size expected to be 2 for the first seed contract",
        "only HCCL reduce_op=sum",
        "bias disabled",
        "float16 only",
        "x1 and x2 fixed to 2D dense ND tensors",
        "PTO seed only covers the local matmul; the benchmark harness performs HCCL all_reduce plus row chunking outside PTODSL",
        "phase-2 rewrite now follows the upstream rank-chunk traversal more closely across the validated 128x256x128 and 64x256x128 shapes",
    ],
}
