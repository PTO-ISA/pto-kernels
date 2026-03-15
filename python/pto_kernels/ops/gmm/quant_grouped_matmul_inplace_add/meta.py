from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="gmm",
    name="quant_grouped_matmul_inplace_add",
    wave="wave1",
    archetype="quant-grouped-matmul-inplace-add",
    ops_transformer_path="gmm/quant_grouped_matmul_inplace_add",
    blockers=[
        "ops-transformer-quant-grouped-matmul-inplace-add-entrypoint-gap",
        "ptodsl-group-routing-primitives",
        "ptodsl-cube-preload-pipeline-surface",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "acl_only_quant_grouped_matmul_inplace_add_probe",
    "shape": {
        "variant_0": {
            "x1": [512, 96],
            "x2": [512, 128],
            "scale1": [4],
            "scale2": [4, 128],
            "y": [4, 96, 128],
            "group_list": [4],
        }
    },
    "limits": [
        "baseline contract probe only",
        "current host exposes ACLNN/C++ coverage but no torch_npu Python entrypoint",
        "PTO kernel implementation is deferred until a reproducible Python or packaged baseline path exists",
    ],
}
