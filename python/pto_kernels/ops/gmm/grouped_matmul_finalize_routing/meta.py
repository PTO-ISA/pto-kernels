from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="gmm",
    name="grouped_matmul_finalize_routing",
    wave="wave1",
    archetype="grouped-matmul-finalize-routing",
    ops_transformer_path="gmm/grouped_matmul_finalize_routing",
    blockers=[
        "ops-transformer-grouped-finalize-routing-format-contract",
        "ptodsl-group-routing-primitives",
        "ptodsl-cube-preload-pipeline-surface",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "w4a8_finalize_routing_contract_probe",
    "shape": {
        "variant_0": {
            "x": [64, 128],
            "weight": [1, 128, 128],
            "scale": [1, 1, 128],
            "pertoken_scale": [64],
            "logit": [64],
            "row_index": [64],
            "group_list": [1],
            "output": [64, 128],
        },
        "variant_1": {
            "x": [64, 192],
            "weight": [1, 192, 1024 // 8],
            "scale": [1, 1, 1024],
            "pertoken_scale": [64],
            "logit": [64],
            "row_index": [64],
            "group_list": [1],
            "output": [64, 1024],
        },
    },
    "limits": [
        "baseline contract probe only",
        "routed quantized path only",
        "host runtime requires scale, pertoken_scale, logit, and row_index",
        "plain dense ND int8 weights are invalid for the probed public runtime path",
        "FRACTAL_NZ weight storage probing is currently blocked by local TBE Python dependency issues",
        "PTO kernel implementation is intentionally deferred until a stable baseline contract is reproducible",
    ],
}
