from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="gmm",
    name="grouped_matmul",
    wave="wave1",
    archetype="gmm",
    ops_transformer_path="gmm/grouped_matmul",
    blockers=[
        "ptodsl-group-routing-primitives",
    ],
)

META["seed_variant"] = {
    "name": "dense_single_weight_bf16_to_bf16",
    "shape": [128, 128, 128],
    "limits": [
        "single batch only",
        "single dense weight only",
        "baseline path uses a single 3D weight tensor to satisfy aclnnGroupedMatmulV5",
        "no bias/quantization/activation",
        "pto path stores ACC output directly to bf16 GM for parity with the baseline contract",
    ],
}
