from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="posembedding",
    name="rotary_position_embedding_grad",
    wave="wave1",
    archetype="rotary-single-grad",
    ops_transformer_path="posembedding/rotary_position_embedding_grad",
    blockers=[
        "ptodsl-rope-generalization",
        "ptodsl-vector-queue-pipeline-surface",
    ],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "fp16_half_rotary_grad_bsnd_bnsd_d128",
    "shape": {
        "variant_0": {
            "layout": "BSND",
            "dy": [2, 32, 1, 128],
            "x": [2, 32, 1, 128],
            "cos": [2, 32, 1, 128],
            "sin": [2, 32, 1, 128],
            "dx": [2, 32, 1, 128],
            "dcos": [2, 32, 1, 128],
            "dsin": [2, 32, 1, 128],
        },
        "variant_1": {
            "layout": "BNSD",
            "dy": [2, 1, 32, 128],
            "x": [2, 1, 32, 128],
            "cos": [2, 1, 32, 128],
            "sin": [2, 1, 32, 128],
            "dx": [2, 1, 32, 128],
            "dcos": [2, 1, 32, 128],
            "dsin": [2, 1, 32, 128],
        },
    },
    "limits": [
        "float16 only",
        "half mode only",
        "head_dim fixed to 128",
        "single-head validated shapes only",
        "xOptional required and constrained to match dy shape exactly",
        "current PTO seed covers the no-reduction dcos/dsin contract only",
        "baseline torch_npu path on this host is parity-checked on dx only because dcos/dsin are returned as zeros",
    ],
}
