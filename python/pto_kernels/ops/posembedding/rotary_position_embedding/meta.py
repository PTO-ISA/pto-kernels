from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="posembedding",
    name="rotary_position_embedding",
    wave="wave1",
    archetype="rotary-single",
    ops_transformer_path="posembedding/rotary_position_embedding",
    blockers=[
        "ptodsl-rope-generalization",
        "ptodsl-vector-queue-pipeline-surface",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "fp16_half_rotary_bsnd_bnsd_d128",
    "shape": {
        "variant_0": {
            "layout": "BSND",
            "x": [2, 32, 1, 128],
            "cos": [2, 32, 1, 128],
            "sin": [2, 32, 1, 128],
            "out": [2, 32, 1, 128],
        },
        "variant_1": {
            "layout": "BNSD",
            "x": [2, 1, 32, 128],
            "cos": [2, 1, 32, 128],
            "sin": [2, 1, 32, 128],
            "out": [2, 1, 32, 128],
        },
    },
    "limits": [
        "float16 only",
        "half mode only",
        "head_dim fixed to 128",
        "single-head validated shapes only",
        "current PTO seed uses flattened [rows, D] row chunks and relies on PTOAS autosync",
    ],
}
