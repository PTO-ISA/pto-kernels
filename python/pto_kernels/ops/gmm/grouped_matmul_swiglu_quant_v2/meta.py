from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="gmm",
    name="grouped_matmul_swiglu_quant_v2",
    wave="wave1",
    archetype="grouped-matmul-swiglu-quant-v2",
    ops_transformer_path="gmm/grouped_matmul_swiglu_quant_v2",
    blockers=[
        "ops-transformer-grouped-swiglu-quant-v2-contract",
        "ptodsl-group-routing-primitives",
        "ptodsl-cube-preload-pipeline-surface",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "fp8_list_weight_swiglu_quant_v2_contract_probe",
    "shape": {
        "variant_0": {
            "x": [2048, 7168],
            "weight_list_item": [8, 7168, 4096],
            "weight_scale_list_item": [8, 112, 4096, 2],
            "x_scale": [2048, 112, 2],
            "group_list": [8],
            "output": [2048, 2048],
            "output_scale": [2048, 32, 2],
        }
    },
    "limits": [
        "baseline contract probe only",
        "requires list-valued low-precision weights and scales",
        "requires FP8-style x/weight/scale contracts from the upstream ACLNN example",
        "PTO kernel implementation is deferred until a stable host baseline slice is reproduced",
    ],
}
