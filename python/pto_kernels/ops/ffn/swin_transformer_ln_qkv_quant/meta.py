"""Metadata for the bounded SwinTransformerLnQkvQuant bring-up slice."""

from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="ffn",
    name="swin_transformer_ln_qkv_quant",
    wave="wave1",
    archetype="swin-transformer-ln-qkv-quant",
    ops_transformer_path="ffn/swin_transformer_ln_qkv_quant",
    blockers=[
        "ops-transformer-swin-ln-qkv-quant-python-entrypoint-gap",
        "ptodsl-swin-ln-qkv-quant-surface",
    ],
)

META["status"] = "blocked"
META["seed_variant"] = {
    "formula": "q/k/v = dequant(int8_matmul(quant(layernorm(x)), weight, bias), dequantScale).split",
    "shape": {
        "smoke": {
            "x": [1, 64, 128],
            "gamma": [128],
            "beta": [128],
            "weight": [128, 384],
            "bias": [384],
            "quantScale": [128],
            "quantOffset": [128],
            "dequantScale": [384],
            "query_output": [1, 4, 64, 32],
            "key_output": [1, 4, 64, 32],
            "value_output": [1, 4, 64, 32],
        },
        "nominal": {
            "x": [8, 64, 128],
            "gamma": [128],
            "beta": [128],
            "weight": [128, 384],
            "bias": [384],
            "quantScale": [128],
            "quantOffset": [128],
            "dequantScale": [384],
            "query_output": [8, 4, 64, 32],
            "key_output": [8, 4, 64, 32],
            "value_output": [8, 4, 64, 32],
        },
        "boundary": {
            "x": [32, 64, 128],
            "gamma": [128],
            "beta": [128],
            "weight": [128, 384],
            "bias": [384],
            "quantScale": [128],
            "quantOffset": [128],
            "dequantScale": [384],
            "query_output": [32, 4, 64, 32],
            "key_output": [32, 4, 64, 32],
            "value_output": [32, 4, 64, 32],
        },
    },
    "limits": [
        "bounded blocked slice only on 910B",
        "public torch_npu baseline entrypoint is unavailable on this host",
        "upstream public docs mark the ACLNN operator unsupported on Atlas A2 / 910B",
        "PTO port is deferred until a real PTODSL/PTOAS/pto-isa quantized LN -> int8 matmul -> dequant path is landed",
        "nominal shape is chosen so a future PTO port can use all requested block ids",
    ],
}
