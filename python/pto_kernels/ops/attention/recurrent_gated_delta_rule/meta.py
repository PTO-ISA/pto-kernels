from pto_kernels.ops.common import planned_meta


META = planned_meta(
    family="attention",
    name="recurrent_gated_delta_rule",
    wave="wave3",
    archetype="attention-recurrent",
    ops_transformer_path="attention/recurrent_gated_delta_rule",
    blockers=["ptoisa-a3-recurrent-state-update-runtime-hang"],
)

META["status"] = "prototype"
META["seed_variant"] = {
    "name": "nd_recurrent_no_gk",
    "shape": {
        "smoke": {
            "query": [2, 16, 16],
            "key": [2, 16, 16],
            "value": [2, 16, 16],
            "beta": [2, 16],
            "g": [2, 16],
            "state": [2, 16, 16, 16],
            "attention_out": [2, 16, 16],
            "final_state": [2, 16, 16, 16],
        },
        "nominal": {
            "query": [4, 16, 64],
            "key": [4, 16, 64],
            "value": [4, 16, 64],
            "beta": [4, 16],
            "g": [4, 16],
            "state": [4, 16, 64, 64],
            "attention_out": [4, 16, 64],
            "final_state": [4, 16, 64, 64],
        },
        "boundary": {
            "query": [8, 16, 128],
            "key": [8, 16, 128],
            "value": [8, 16, 128],
            "beta": [8, 16],
            "g": [8, 16],
            "state": [8, 16, 128, 128],
            "attention_out": [8, 16, 128],
            "final_state": [8, 16, 128, 128],
        },
    },
    "limits": [
        "constrained ND recurrent slice with batch=1 and nv=nk=16",
        "q/k/v/state/beta use bfloat16; g uses float32; gk fixed to None",
        "actual_seq_lengths fixed to the full sequence length, ssm_state_indices fixed to arange(T)",
        "num_accepted_tokens fixed to ones",
    ],
}
