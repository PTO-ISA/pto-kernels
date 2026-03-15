"""Runtime helpers for the first moe_re_routing migration slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch_npu  # noqa: F401


@dataclass(frozen=True)
class MoeReRoutingVariant:
    counts: tuple[tuple[int, ...], ...]
    hidden_size: int
    seed: int = 0
    dtype: str = "float16"
    use_scales: bool = True
    expert_token_num_type: int = 1
    idx_type: int = 0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def ranks(self) -> int:
        return len(self.counts)

    @property
    def experts(self) -> int:
        return len(self.counts[0])

    @property
    def tokens(self) -> int:
        return sum(sum(row) for row in self.counts)

    @property
    def label(self) -> str:
        return f"t{self.tokens}_h{self.hidden_size}_r{self.ranks}_e{self.experts}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "tokens": [self.tokens, self.hidden_size],
            "expert_token_num_per_rank": [self.ranks, self.experts],
            "per_token_scales": [self.tokens],
            "permute_tokens": [self.tokens, self.hidden_size],
            "permute_per_token_scales": [self.tokens],
            "permute_token_idx": [self.tokens],
            "expert_token_num": [self.experts],
        }


VARIANT = MoeReRoutingVariant(
    counts=((2, 1), (3, 2)),
    hidden_size=16,
    seed=0,
)
VARIANTS = (
    VARIANT,
    MoeReRoutingVariant(
        counts=((20, 12, 16, 16), (18, 14, 10, 22), (12, 20, 18, 14), (14, 18, 12, 20)),
        hidden_size=64,
        seed=1,
    ),
    MoeReRoutingVariant(
        counts=((16, 0, 8, 8), (0, 20, 4, 8), (12, 4, 24, 0), (8, 4, 0, 12)),
        hidden_size=128,
        seed=2,
    ),
)


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _reference_outputs(
    tokens_cpu: torch.Tensor,
    counts_cpu: torch.Tensor,
    scales_cpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ranks, experts = counts_cpu.shape
    src_offsets: dict[tuple[int, int], int] = {}
    src_cursor = 0
    for rank in range(ranks):
        for expert in range(experts):
            src_offsets[(rank, expert)] = src_cursor
            src_cursor += int(counts_cpu[rank, expert].item())

    permuted_tokens = []
    permuted_scales = []
    permute_idx = []
    expert_token_num = []

    for expert in range(experts):
        expert_total = 0
        for rank in range(ranks):
            count = int(counts_cpu[rank, expert].item())
            expert_total += count
            src_base = src_offsets[(rank, expert)]
            for token_offset in range(count):
                src_idx = src_base + token_offset
                permuted_tokens.append(tokens_cpu[src_idx])
                permuted_scales.append(scales_cpu[src_idx])
                permute_idx.append(src_idx)
        expert_token_num.append(expert_total)

    return (
        torch.stack(permuted_tokens).float(),
        torch.stack(permuted_scales).float(),
        torch.tensor(permute_idx, dtype=torch.int32),
        torch.tensor(expert_token_num, dtype=counts_cpu.dtype),
    )


def make_moe_re_routing_inputs(
    variant: MoeReRoutingVariant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(variant.seed)

    tokens_cpu = torch.randn(
        (variant.tokens, variant.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float16)
    counts_cpu = torch.tensor(variant.counts, dtype=torch.int32)
    scales_cpu = torch.randn((variant.tokens,), generator=generator, dtype=torch.float32)

    (
        ref_tokens,
        ref_scales,
        ref_idx,
        ref_expert_token_num,
    ) = _reference_outputs(tokens_cpu, counts_cpu, scales_cpu)

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "tokens": tokens_cpu.npu(),
        "expert_token_num_per_rank": counts_cpu.npu(),
        "per_token_scales": scales_cpu.npu(),
        "permute_tokens_pto": torch.empty_like(tokens_cpu).npu(),
        "permute_per_token_scales_pto": torch.empty_like(scales_cpu).npu(),
        "permute_token_idx_pto": torch.empty((variant.tokens,), dtype=torch.int32).npu(),
        "expert_token_num_pto": torch.empty((variant.experts,), dtype=torch.int32).npu(),
        "reference_permute_tokens": ref_tokens,
        "reference_permute_per_token_scales": ref_scales,
        "reference_permute_token_idx": ref_idx,
        "reference_expert_token_num": ref_expert_token_num,
    }


def run_torch_npu_moe_re_routing(inputs: dict[str, object]):
    return torch_npu.npu_moe_re_routing(
        inputs["tokens"],
        inputs["expert_token_num_per_rank"],
        per_token_scales=inputs["per_token_scales"],
        expert_token_num_type=inputs["variant"]["expert_token_num_type"],
        idx_type=inputs["variant"]["idx_type"],
    )


def run_pto_moe_re_routing_variant(wrapper, inputs: dict[str, object]):
    wrapper(
        inputs["permute_tokens_pto"],
        inputs["permute_per_token_scales_pto"],
        inputs["permute_token_idx_pto"],
        inputs["expert_token_num_pto"],
        inputs["tokens"],
        inputs["expert_token_num_per_rank"],
        inputs["per_token_scales"],
    )
    return (
        inputs["permute_tokens_pto"].float(),
        inputs["permute_per_token_scales_pto"].float(),
        inputs["permute_token_idx_pto"].cpu(),
        inputs["expert_token_num_pto"].cpu(),
    )
