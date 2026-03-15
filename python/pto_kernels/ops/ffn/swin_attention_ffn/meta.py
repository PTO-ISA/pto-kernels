"""Metadata for the constrained SwinAttentionFFN PTO slice."""

META = {
    "name": "swin_attention_ffn",
    "family": "ffn",
    "wave": "wave1",
    "phase": "phase3",
    "status": "prototype",
    "seed_variant": {
        "formula": "y = x1 @ x2 + bias + x3",
        "constraints": [
            "910B zero-shift slice only (shift1=0, shift2=0)",
            "fp16 ND inputs and outputs",
            "x1 and x3 flattened to [tokens, hidden]",
            "nominal variant uses all block ids",
        ],
        "smoke_shape": {"x1": [2, 64, 128], "x2": [128, 128], "bias": [128], "x3": [2, 64, 128]},
        "nominal_shape": {"x1": [48, 64, 128], "x2": [128, 128], "bias": [128], "x3": [48, 64, 128]},
        "boundary_shape": {"x1": [8, 64, 128], "x2": [128, 128], "bias": [128], "x3": [8, 64, 128]},
    },
    "tiling": {
        "base_m": 128,
        "base_n": 128,
        "base_k": 128,
        "block_dim": 24,
        "add_block_dim": 24,
    },
    "blockers": [
        "ops-transformer-swin-ffn-python-entrypoint-gap",
        "ptodsl-swin-layout-shift-surface",
    ],
}
