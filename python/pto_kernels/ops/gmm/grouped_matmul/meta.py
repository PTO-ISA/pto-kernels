from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="gmm",
    name="grouped_matmul",
    wave="wave1",
    archetype="gmm",
    ops_transformer_path="gmm/grouped_matmul",
    blockers=[
        "ptodsl-bf16-epilogue-conversion",
        "ptodsl-group-routing-primitives",
        "ops-transformer-runtime-package-bringup",
    ],
)

META["seed_variant"] = {
    "name": "dense_single_weight_bf16_to_f32",
    "shape": [128, 128, 128],
    "limits": [
        "single batch only",
        "single dense weight only",
        "no bias/quantization/activation",
        "output kept in float32 until bf16 epilogue lands in ptodsl",
    ],
}
