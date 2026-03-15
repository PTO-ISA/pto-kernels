from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="attention",
    name="attention_update",
    wave="wave3",
    archetype="attention-update",
    ops_transformer_path="attention/attention_update",
    blockers=[],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "sp2_fp16_update_type0",
    "shape": {
        "smoke": {
            "lse0": [8],
            "lse1": [8],
            "local_out0": [8, 16],
            "local_out1": [8, 16],
            "out": [8, 16],
        },
        "nominal": {
            "lse0": [256],
            "lse1": [256],
            "local_out0": [256, 64],
            "local_out1": [256, 64],
            "out": [256, 64],
        },
        "boundary": {
            "lse0": [128],
            "lse1": [128],
            "local_out0": [128, 128],
            "local_out1": [128, 128],
            "out": [128, 128],
        },
    },
    "limits": [
        "constrained sp=2 slice only",
        "updateType fixed to 0 so lseOut is omitted",
        "float32 lse inputs, float16 localOut/output",
        "row-wise update contract only",
        "nominal shape uses all requested block ids on the checked-in block_dim",
    ],
}
