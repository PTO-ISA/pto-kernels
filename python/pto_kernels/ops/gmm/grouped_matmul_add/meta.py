from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="gmm",
    name="grouped_matmul_add",
    wave="wave1",
    archetype="grouped-matmul-add",
    ops_transformer_path="gmm/grouped_matmul_add",
    blockers=[
        "ptodsl-cube-preload-pipeline-surface",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "dense_single_group_bf16_input_f32_accumulate_add",
    "shape": {
        "variant_0": {
            "x_t": [128, 64],
            "x_pto": [64, 128],
            "weight": [128, 128],
            "y_init": [64, 128],
            "output": [64, 128],
        },
        "variant_1": {
            "x_t": [128, 128],
            "x_pto": [128, 128],
            "weight": [128, 256],
            "y_init": [128, 256],
            "output": [128, 256],
        },
    },
    "limits": [
        "single dense group only",
        "bf16 x/weight only",
        "f32 y/output only",
        "transposeX baseline contract only",
        "transposeWeight fixed to false",
        "groupType fixed to 2 (K-axis grouping)",
        "PTO path currently materializes the baseline-format x_t into x_pto = x_t^T before entering the PTO kernel",
    ],
}
