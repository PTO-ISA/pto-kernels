from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="gmm",
    name="grouped_matmul_swiglu_quant",
    wave="wave1",
    archetype="grouped-matmul-swiglu-quant",
    ops_transformer_path="gmm/grouped_matmul_swiglu_quant",
    blockers=[
        "ops-transformer-grouped-swiglu-quant-format-contract",
        "ptodsl-group-routing-primitives",
        "ptodsl-cube-preload-pipeline-surface",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "fractal_nz_int8_swiglu_quant_contract_probe",
    "shape": {
        "variant_0": {
            "x": [1024, 256],
            "weight_logical": [16, 256, 4096],
            "weight_storage_nz": [16, 128, 16, 16, 32],
            "weight_scale": [16, 4096],
            "x_scale": [1024],
            "group_list": [16],
            "output": [1024, 2048],
            "output_scale": [1024],
        }
    },
    "limits": [
        "baseline contract probe only",
        "requires FRACTAL_NZ weight storage metadata",
        "requires int8 input/output quantized path",
        "PTO kernel implementation is deferred until the host baseline contract is reproducible",
    ],
}
