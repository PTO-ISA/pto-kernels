from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="ffn",
    name="ffn",
    wave="wave1",
    archetype="ffn",
    ops_transformer_path="ffn/ffn",
    blockers=[
        "ptodsl-ffn-interstage-cast-path",
    ],
)

META["seed_variant"] = {
    "name": "dense_relu_fp16",
    "shape": [32, 128, 256, 128],
    "limits": [
        "dense FFN only, no expert tokens",
        "activation fixed to relu",
        "float16 only",
        "bias disabled",
        "seed tensors scaled by 0.125 for stable fp16 reference comparison",
        "PTO seed currently runs as a staged matmul -> relu -> matmul pipeline",
    ],
}
