"""Runtime helpers for the first recurrent_gated_delta_rule migration slice."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class RecurrentGatedDeltaRuleVariant:
    seq_len: int
    dim: int
    seed: int
    batch_size: int = 1
    num_heads: int = 16
    num_value_heads: int = 16
    input_scale: float = 0.125

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"t{self.seq_len}_h{self.num_heads}_d{self.dim}"

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.dim)

    @property
    def total_tokens(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "query": [self.total_tokens, self.num_heads, self.dim],
            "key": [self.total_tokens, self.num_heads, self.dim],
            "value": [self.total_tokens, self.num_value_heads, self.dim],
            "beta": [self.total_tokens, self.num_value_heads],
            "g": [self.total_tokens, self.num_value_heads],
            "state": [self.total_tokens, self.num_value_heads, self.dim, self.dim],
            "attention_out": [self.total_tokens, self.num_value_heads, self.dim],
            "final_state": [self.total_tokens, self.num_value_heads, self.dim, self.dim],
        }


VARIANT = RecurrentGatedDeltaRuleVariant(seq_len=2, dim=16, seed=0)
VARIANTS = (
    RecurrentGatedDeltaRuleVariant(seq_len=2, dim=16, seed=0),
    RecurrentGatedDeltaRuleVariant(seq_len=4, dim=64, seed=1),
    RecurrentGatedDeltaRuleVariant(seq_len=8, dim=128, seed=2),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _cpu_reference(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = query.float()
    k = key.float()
    v = value.float()
    s = state.float().clone()
    beta_f = beta.float()
    g_f = g.float()
    idx = ssm_state_indices.to(torch.int64)

    total_tokens = q.shape[0]
    num_value_heads = v.shape[1]
    num_heads = q.shape[1]
    group_size = num_value_heads // num_heads

    out = torch.empty((total_tokens, num_value_heads, v.shape[2]), dtype=torch.float32)
    final_state = s.clone()
    for token_idx in range(total_tokens):
        state_slot = int(idx[token_idx].item())
        for value_head in range(num_value_heads):
            key_head = value_head // group_size
            q_vec = q[token_idx, key_head]
            k_vec = k[token_idx, key_head]
            v_vec = v[token_idx, value_head]
            s_prev = final_state[state_slot, value_head]
            alpha = torch.exp(g_f[token_idx, value_head])
            proj = s_prev @ k_vec
            delta = v_vec - alpha * proj
            s_new = alpha * s_prev + beta_f[token_idx, value_head] * torch.outer(delta, k_vec)
            out[token_idx, value_head] = (s_new @ q_vec) * scale
            final_state[state_slot, value_head] = s_new

    return out.to(torch.bfloat16), final_state.to(torch.bfloat16)


def make_recurrent_gated_delta_rule_inputs(
    variant: RecurrentGatedDeltaRuleVariant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    g = torch.Generator(device="cpu")
    g.manual_seed(variant.seed)

    total_tokens = variant.total_tokens
    query_cpu = (
        torch.randn((total_tokens, variant.num_heads, variant.dim), generator=g, dtype=torch.float32)
        .mul(variant.input_scale)
        .to(torch.bfloat16)
    )
    key_cpu = (
        torch.randn((total_tokens, variant.num_heads, variant.dim), generator=g, dtype=torch.float32)
        .mul(variant.input_scale)
        .to(torch.bfloat16)
    )
    query_cpu = F.normalize(query_cpu.float(), p=2.0, dim=-1).to(torch.bfloat16)
    key_cpu = F.normalize(key_cpu.float(), p=2.0, dim=-1).to(torch.bfloat16)
    value_cpu = (
        torch.randn((total_tokens, variant.num_value_heads, variant.dim), generator=g, dtype=torch.float32)
        .mul(variant.input_scale)
        .to(torch.bfloat16)
    )
    state_cpu = (
        torch.randn(
            (total_tokens, variant.num_value_heads, variant.dim, variant.dim),
            generator=g,
            dtype=torch.float32,
        )
        .mul(variant.input_scale)
        .to(torch.bfloat16)
    )
    beta_cpu = torch.rand((total_tokens, variant.num_value_heads), generator=g, dtype=torch.float32).to(torch.bfloat16)
    g_cpu = (torch.rand((total_tokens, variant.num_value_heads), generator=g, dtype=torch.float32) * -0.1).to(torch.float32)
    actual_seq_lengths_cpu = torch.tensor([variant.seq_len] * variant.batch_size, dtype=torch.int32)
    ssm_state_indices_cpu = torch.arange(total_tokens, dtype=torch.int32)
    num_accepted_tokens_cpu = torch.ones((variant.batch_size,), dtype=torch.int32)

    reference_out, reference_state = _cpu_reference(
        query=query_cpu,
        key=key_cpu,
        value=value_cpu,
        state=state_cpu,
        beta=beta_cpu,
        g=g_cpu,
        ssm_state_indices=ssm_state_indices_cpu,
        scale=variant.scale,
    )

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "query": query_cpu.npu(),
        "key": key_cpu.npu(),
        "value": value_cpu.npu(),
        "state": state_cpu.clone().npu(),
        "state_functional": state_cpu.clone().npu(),
        "beta": beta_cpu.npu(),
        "g": g_cpu.npu(),
        "actual_seq_lengths": actual_seq_lengths_cpu.npu(),
        "ssm_state_indices": ssm_state_indices_cpu.npu(),
        "num_accepted_tokens": num_accepted_tokens_cpu.npu(),
        "reference_out": reference_out,
        "reference_state": reference_state,
    }


def run_torch_npu_recurrent_gated_delta_rule(inputs: dict[str, object]):
    variant = inputs["variant"]
    return torch_npu.npu_recurrent_gated_delta_rule(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        inputs["state"],
        beta=inputs["beta"],
        scale=1.0 / math.sqrt(variant["dim"]),
        actual_seq_lengths=inputs["actual_seq_lengths"],
        ssm_state_indices=inputs["ssm_state_indices"],
        num_accepted_tokens=inputs["num_accepted_tokens"],
        g=inputs["g"],
        gk=None,
    )


def run_torch_npu_recurrent_gated_delta_rule_functional(inputs: dict[str, object]):
    variant = inputs["variant"]
    return torch_npu.npu_recurrent_gated_delta_rule_functional(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        inputs["state_functional"],
        beta=inputs["beta"],
        scale=1.0 / math.sqrt(variant["dim"]),
        actual_seq_lengths=inputs["actual_seq_lengths"],
        ssm_state_indices=inputs["ssm_state_indices"],
        num_accepted_tokens=inputs["num_accepted_tokens"],
        g=inputs["g"],
        gk=None,
    )
