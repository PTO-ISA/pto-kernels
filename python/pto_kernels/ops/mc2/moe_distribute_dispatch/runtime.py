"""Runtime helpers for the phase-2 moe_distribute_dispatch seed."""

from __future__ import annotations

import importlib.util
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

from pto_kernels.utils import run_local_ranked_job


SUPPORTED_WORLD_SIZES = (8,)


@dataclass(frozen=True)
class MoeDistributeDispatchVariant:
    tokens: int = 8
    hidden_size: int = 7168
    dtype: str = "float16"
    expected_world_size: int = 8
    topk: int = 1
    quant_mode: int = 0
    seed: int = 0
    input_scale: float = 0.125

    def as_dict(self) -> dict[str, int | str | float]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"t{self.tokens}_h{self.hidden_size}_w{self.expected_world_size}"

    @property
    def global_bs(self) -> int:
        return self.tokens * self.expected_world_size

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x_local": [self.tokens, self.hidden_size],
            "expert_ids_local": [self.tokens, self.topk],
            "send_buffer_local": [self.tokens, self.hidden_size],
            "expand_x": [self.global_bs, self.hidden_size],
            "expand_idx": [self.tokens * self.topk],
            "expert_token_nums": [1],
            "ep_recv_counts_prefix": [self.expected_world_size],
            "world_size": self.expected_world_size,
        }


VARIANT = MoeDistributeDispatchVariant()
VARIANTS = (
    MoeDistributeDispatchVariant(tokens=8, hidden_size=7168, expected_world_size=8, seed=0),
)


def launcher_blocker(world_size: int) -> dict[str, object] | None:
    enabled = os.environ.get("PTO_ENABLE_UNSTABLE_8RANK_MC2") == "1"
    if world_size == 8 and not enabled:
        return {
            "status": "blocked",
            "reason": (
                "The current local HCCL launcher path is not yet stable for the required 8-rank "
                "A2 MC2 contract. Set PTO_ENABLE_UNSTABLE_8RANK_MC2=1 to attempt the benchmark anyway."
            ),
        }
    return None


def _resolve_variant(
    variant: MoeDistributeDispatchVariant | None = None,
) -> MoeDistributeDispatchVariant:
    return VARIANT if variant is None else variant


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _make_rank_tensors(
    rank: int,
    variant: MoeDistributeDispatchVariant | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved = _resolve_variant(variant)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(resolved.seed + rank)
    x = (
        torch.randn((resolved.tokens, resolved.hidden_size), generator=generator, dtype=torch.float32)
        * resolved.input_scale
    ).to(torch.float16)

    # Keep the first slice deterministic and balanced while still varying per rank/variant.
    base = torch.arange(resolved.tokens, dtype=torch.int64)
    route = ((base + rank + resolved.seed) % resolved.expected_world_size).to(torch.int32)
    expert_ids = route.unsqueeze(1).contiguous()
    return x, expert_ids


def _bincount_destinations(expert_ids: torch.Tensor, world_size: int) -> torch.Tensor:
    flat = expert_ids.reshape(-1).to(torch.int64)
    return torch.bincount(flat, minlength=world_size).to(torch.int32)


def _expand_idx_for_local_tokens(expert_ids: torch.Tensor, world_size: int) -> torch.Tensor:
    del world_size
    counts: dict[int, int] = {}
    values = []
    for expert in expert_ids.reshape(-1).tolist():
        idx = counts.get(int(expert), 0)
        values.append(idx)
        counts[int(expert)] = idx + 1
    return torch.tensor(values, dtype=torch.int32)


def _send_order(expert_ids: torch.Tensor, world_size: int) -> torch.Tensor:
    groups = []
    flat = expert_ids.reshape(-1)
    for dest in range(world_size):
        positions = torch.nonzero(flat == dest, as_tuple=False).reshape(-1)
        if positions.numel() > 0:
            groups.append(positions.to(torch.int64))
    if groups:
        return torch.cat(groups, dim=0)
    return torch.empty((0,), dtype=torch.int64)


def _gather_map(send_order: torch.Tensor, hidden_size: int) -> torch.Tensor:
    if send_order.numel() == 0:
        return torch.empty((0,), dtype=torch.int32)
    column = torch.arange(hidden_size, dtype=torch.int64).unsqueeze(0)
    gather = send_order.unsqueeze(1) * hidden_size + column
    return gather.reshape(-1).to(torch.int32)


def _reference_outputs_for_rank(
    rank: int,
    world_size: int,
    variant: MoeDistributeDispatchVariant | None = None,
) -> dict[str, torch.Tensor]:
    resolved = _resolve_variant(variant)
    local_x, local_expert_ids = _make_rank_tensors(rank, resolved)
    local_expand_idx = _expand_idx_for_local_tokens(local_expert_ids, world_size)

    recv_counts = []
    recv_chunks = []
    for src_rank in range(world_size):
        src_x, src_expert_ids = _make_rank_tensors(src_rank, resolved)
        mask = src_expert_ids.reshape(-1) == rank
        recv_counts.append(int(mask.sum().item()))
        if int(mask.sum().item()) > 0:
            recv_chunks.append(src_x.index_select(0, torch.nonzero(mask, as_tuple=False).reshape(-1)))

    total_rows = sum(recv_counts)
    expand_x = torch.zeros((resolved.global_bs, resolved.hidden_size), dtype=torch.float16)
    if recv_chunks:
        stacked = torch.cat(recv_chunks, dim=0)
        expand_x[: stacked.shape[0]].copy_(stacked)

    return {
        "expand_x": expand_x,
        "expand_idx": local_expand_idx,
        "expert_token_nums": torch.tensor([total_rows], dtype=torch.int64),
        "ep_recv_counts_prefix": torch.tensor(recv_counts, dtype=torch.int32),
        "valid_rows": torch.tensor(total_rows, dtype=torch.int64),
    }


def make_local_pack_inputs(
    rank: int,
    variant: MoeDistributeDispatchVariant | None = None,
    *,
    device: str,
) -> dict[str, object]:
    resolved = _resolve_variant(variant)
    x_cpu, expert_ids_cpu = _make_rank_tensors(rank, resolved)
    send_order_cpu = _send_order(expert_ids_cpu, resolved.expected_world_size)
    gather_indices_cpu = _gather_map(send_order_cpu, resolved.hidden_size)
    send_counts_cpu = _bincount_destinations(expert_ids_cpu, resolved.expected_world_size)
    return {
        "variant": resolved.as_dict(),
        "shape_summary": resolved.shape_summary,
        "x": x_cpu.to(device),
        "expert_ids": expert_ids_cpu.to(device),
        "gather_indices": gather_indices_cpu.to(device),
        "send_buffer": torch.empty_like(x_cpu).to(device),
        "send_counts": send_counts_cpu.to(device),
        "reference": _reference_outputs_for_rank(rank, resolved.expected_world_size, resolved),
    }


def baseline_blocker(*, device_index: int) -> dict[str, object]:
    symbol_available = hasattr(torch_npu, "npu_moe_distribute_dispatch")
    npu_available = torch.npu.is_available()
    detected_npus = torch.npu.device_count() if npu_available else 0
    environment = {
        "device_index": device_index,
        "npu_available": bool(npu_available),
        "detected_npus": int(detected_npus),
        "symbol_available": bool(symbol_available),
    }
    if not symbol_available:
        return {
            "status": "blocked",
            "reason": "torch_npu.npu_moe_distribute_dispatch is unavailable on this host.",
            "environment": environment,
        }
    if detected_npus < VARIANT.expected_world_size:
        return {
            "status": "blocked",
            "reason": (
                f"Need {VARIANT.expected_world_size} NPUs for moe_distribute_dispatch bring-up, "
                f"but only {detected_npus} detected."
            ),
            "environment": environment,
        }
    return {
        "status": "ready",
        "environment": environment,
        "entrypoint": "torch_npu.npu_moe_distribute_dispatch",
    }


def _get_hccl_comm_name(rank: int) -> str:
    from torch.distributed.distributed_c10d import _get_default_group

    default_pg = _get_default_group()
    try:
        backend = default_pg._get_backend(torch.device("npu"))
        return backend.get_hccl_comm_name(rank)
    except AttributeError:
        return default_pg.get_hccl_comm_name(rank)


def _call_baseline_dispatch(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    hcom_name: str,
    world_size: int,
    rank: int,
    variant: MoeDistributeDispatchVariant,
):
    return torch_npu.npu_moe_distribute_dispatch(
        x=x,
        expert_ids=expert_ids,
        group_ep=hcom_name,
        ep_world_size=world_size,
        ep_rank_id=rank,
        moe_expert_num=world_size,
        scales=None,
        x_active_mask=None,
        expert_scales=None,
        group_tp="",
        tp_world_size=0,
        tp_rank_id=0,
        expert_shard_type=0,
        shared_expert_num=0,
        shared_expert_rank_num=0,
        quant_mode=variant.quant_mode,
        global_bs=variant.global_bs,
        expert_token_nums_type=1,
    )


def _extract_baseline_outputs(
    outputs,
    *,
    rank: int,
    world_size: int,
    variant: MoeDistributeDispatchVariant,
) -> dict[str, torch.Tensor]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 5:
        raise RuntimeError(
            "npu_moe_distribute_dispatch returned an unexpected output signature; "
            f"expected at least 5 values, got {type(outputs).__name__}."
        )
    reference = _reference_outputs_for_rank(rank, world_size, variant)
    expand_x = outputs[0]
    expand_idx = outputs[2]
    expert_token_nums = outputs[3]
    ep_recv_counts = outputs[4]
    return {
        "expand_x": expand_x,
        "expand_idx": expand_idx,
        "expert_token_nums": expert_token_nums,
        "ep_recv_counts_prefix": ep_recv_counts.reshape(-1)[: reference["ep_recv_counts_prefix"].numel()],
        "valid_rows": reference["valid_rows"].clone(),
    }


def _correctness_metrics(
    actual: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> dict[str, float]:
    valid_rows = int(reference["valid_rows"].item())
    actual_expand = actual["expand_x"][:valid_rows].float().cpu()
    ref_expand = reference["expand_x"][:valid_rows].float()
    return {
        "expand_x_max_abs_diff": (actual_expand - ref_expand).abs().max().item() if valid_rows > 0 else 0.0,
        "expand_idx_max_abs_diff": (
            actual["expand_idx"].reshape(-1)[: reference["expand_idx"].numel()].to(torch.int32).cpu()
            - reference["expand_idx"]
        )
        .abs()
        .max()
        .item(),
        "expert_token_nums_max_abs_diff": (
            actual["expert_token_nums"].reshape(-1)[: reference["expert_token_nums"].numel()].to(torch.int64).cpu()
            - reference["expert_token_nums"]
        )
        .abs()
        .max()
        .item(),
        "ep_recv_counts_prefix_max_abs_diff": (
            actual["ep_recv_counts_prefix"].reshape(-1).to(torch.int32).cpu()
            - reference["ep_recv_counts_prefix"]
        )
        .abs()
        .max()
        .item(),
    }


def _baseline_worker(
    *,
    rank: int,
    world_size: int,
    output_dir: Path,
    device: str,
    warmup: int,
    repeat: int,
    variant_dict: dict[str, object],
):
    del output_dir
    variant = MoeDistributeDispatchVariant(**variant_dict)
    inputs = make_local_pack_inputs(rank, variant, device=device)
    x = inputs["x"]
    expert_ids = inputs["expert_ids"]
    reference = inputs["reference"]
    hcom_name = _get_hccl_comm_name(rank)

    for _ in range(warmup):
        dist.barrier()
        _call_baseline_dispatch(x, expert_ids, hcom_name=hcom_name, world_size=world_size, rank=rank, variant=variant)
    torch.npu.synchronize()
    dist.barrier()

    timings_ms: list[float] = []
    actual = None
    for _ in range(repeat):
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        outputs = _call_baseline_dispatch(
            x, expert_ids, hcom_name=hcom_name, world_size=world_size, rank=rank, variant=variant
        )
        torch.npu.synchronize()
        dist.barrier()
        timings_ms.append((time.perf_counter() - start) * 1000.0)
        actual = _extract_baseline_outputs(outputs, rank=rank, world_size=world_size, variant=variant)

    if actual is None:
        raise RuntimeError("npu_moe_distribute_dispatch did not return output tensors.")

    metrics = _correctness_metrics(actual, reference)
    return {
        "status": "ok",
        "rank": rank,
        "world_size": world_size,
        "device": device,
        "hcom_name": hcom_name,
        "timings_ms": {
            "median": statistics.median(timings_ms),
            "min": min(timings_ms),
            "max": max(timings_ms),
        },
        "correctness": {
            **metrics,
            "max_abs_diff": max(metrics.values()),
        },
    }


def run_distributed_baseline_benchmark(
    *,
    variant: MoeDistributeDispatchVariant = VARIANT,
    artifacts_dir: Path,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    world_size = int(os.environ.get("PTO_MC2_MOE_DISPATCH_WORLD_SIZE", variant.expected_world_size))
    if world_size not in SUPPORTED_WORLD_SIZES:
        return {
            "status": "blocked",
            "reason": f"Unsupported world size {world_size}; expected one of {SUPPORTED_WORLD_SIZES}.",
            "variant": variant.as_dict(),
        }
    blocker = launcher_blocker(world_size)
    if blocker is not None:
        blocker["variant"] = variant.as_dict()
        return blocker

    output_dir = Path(artifacts_dir) / "distributed_baseline"
    launch = run_local_ranked_job(
        _baseline_worker,
        world_size=world_size,
        output_dir=output_dir,
        worker_kwargs={"warmup": warmup, "repeat": repeat, "variant_dict": variant.as_dict()},
    )
    if launch["status"] != "ok":
        return {
            "status": "blocked",
            "reason": launch.get("reason", "Distributed moe_distribute_dispatch baseline failed."),
            "variant": variant.as_dict(),
            "world_size": world_size,
            "rank_reports": launch.get("rank_reports", []),
        }

    rank_reports = launch["rank_reports"]
    per_rank_medians = [report["timings_ms"]["median"] for report in rank_reports]
    max_abs_diff = max(report["correctness"]["max_abs_diff"] for report in rank_reports)
    return {
        "status": "ok",
        "variant": variant.as_dict(),
        "entrypoint": "torch_npu.npu_moe_distribute_dispatch",
        "world_size": world_size,
        "shape_summary": variant.shape_summary,
        "timings_ms": {
            "median": max(per_rank_medians),
            "min": min(report["timings_ms"]["min"] for report in rank_reports),
            "max": max(report["timings_ms"]["max"] for report in rank_reports),
            "per_rank_median": per_rank_medians,
        },
        "correctness": {
            "max_abs_diff": max_abs_diff,
            "per_rank_max_abs_diff": [report["correctness"]["max_abs_diff"] for report in rank_reports],
        },
        "reference_contract": "ep_only_top1_no_quant_dispatch",
        "rank_reports": rank_reports,
    }


def _load_kernel_module():
    kernel_path = Path(__file__).with_name("kernel.py")
    spec = importlib.util.spec_from_file_location("pto_mc2_moe_distribute_dispatch_kernel", kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import PTO kernel module from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_gather_send_counts(local_counts: torch.Tensor, world_size: int) -> torch.Tensor:
    gathered = [torch.empty_like(local_counts) for _ in range(world_size)]
    dist.all_gather(gathered, local_counts)
    return torch.stack(gathered, dim=0)


def _pto_worker(
    *,
    rank: int,
    world_size: int,
    output_dir: Path,
    device: str,
    warmup: int,
    repeat: int,
    variant_dict: dict[str, object],
):
    variant = MoeDistributeDispatchVariant(**variant_dict)
    os.environ["PTO_MC2_MOE_DISPATCH_WORLD_SIZE"] = str(world_size)
    module = _load_kernel_module()
    wrapper = module.build_jit_wrapper(output_dir=output_dir / f"rank_{rank}_kernel")
    build = getattr(wrapper, "_build", None)
    if callable(build):
        build()

    inputs = make_local_pack_inputs(rank, variant, device=device)
    x = inputs["x"]
    gather_indices = inputs["gather_indices"]
    send_buffer = inputs["send_buffer"]
    send_counts = inputs["send_counts"].to(torch.int64)
    reference = inputs["reference"]

    all_send_counts = _all_gather_send_counts(send_counts, world_size)
    recv_counts = all_send_counts[:, rank].contiguous()
    send_splits = [int(item) * variant.hidden_size for item in send_counts.cpu().tolist()]
    recv_splits = [int(item) * variant.hidden_size for item in recv_counts.cpu().tolist()]
    recv_buffer = torch.zeros((variant.global_bs, variant.hidden_size), dtype=torch.float16).to(device)

    def _run_once():
        wrapper(send_buffer, x, gather_indices)
        recv_buffer.zero_()
        dist.all_to_all_single(
            recv_buffer.reshape(-1),
            send_buffer.reshape(-1),
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        return {
            "expand_x": recv_buffer,
            "expand_idx": reference["expand_idx"].to(device),
            "expert_token_nums": reference["expert_token_nums"].to(device),
            "ep_recv_counts_prefix": recv_counts.to(torch.int32),
            "valid_rows": reference["valid_rows"].to(device),
        }

    for _ in range(warmup):
        dist.barrier()
        _run_once()
    torch.npu.synchronize()
    dist.barrier()

    timings_ms: list[float] = []
    actual = None
    for _ in range(repeat):
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        actual = _run_once()
        torch.npu.synchronize()
        dist.barrier()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    if actual is None:
        raise RuntimeError("PTO moe_distribute_dispatch worker produced no output.")

    metrics = _correctness_metrics(actual, reference)
    return {
        "status": "ok",
        "rank": rank,
        "world_size": world_size,
        "device": device,
        "timings_ms": {
            "median": statistics.median(timings_ms),
            "min": min(timings_ms),
            "max": max(timings_ms),
        },
        "correctness": {
            **metrics,
            "max_abs_diff": max(metrics.values()),
        },
        "reference_contract": "host_precomputed_send_order_then_all_to_all_single",
    }


def run_distributed_pto_benchmark(
    *,
    variant: MoeDistributeDispatchVariant = VARIANT,
    artifacts_dir: Path,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    world_size = int(os.environ.get("PTO_MC2_MOE_DISPATCH_WORLD_SIZE", variant.expected_world_size))
    if world_size not in SUPPORTED_WORLD_SIZES:
        return {
            "status": "blocked",
            "reason": f"Unsupported world size {world_size}; expected one of {SUPPORTED_WORLD_SIZES}.",
            "variant": variant.as_dict(),
        }
    blocker = launcher_blocker(world_size)
    if blocker is not None:
        blocker["variant"] = variant.as_dict()
        return blocker

    output_dir = Path(artifacts_dir) / "distributed_pto"
    launch = run_local_ranked_job(
        _pto_worker,
        world_size=world_size,
        output_dir=output_dir,
        worker_kwargs={"warmup": warmup, "repeat": repeat, "variant_dict": variant.as_dict()},
    )
    if launch["status"] != "ok":
        return {
            "status": "blocked",
            "reason": launch.get("reason", "Distributed PTO moe_distribute_dispatch failed."),
            "variant": variant.as_dict(),
            "world_size": world_size,
            "rank_reports": launch.get("rank_reports", []),
        }

    rank_reports = launch["rank_reports"]
    per_rank_medians = [report["timings_ms"]["median"] for report in rank_reports]
    max_abs_diff = max(report["correctness"]["max_abs_diff"] for report in rank_reports)
    return {
        "status": "ok",
        "variant": variant.as_dict(),
        "world_size": world_size,
        "shape_summary": variant.shape_summary,
        "timings_ms": {
            "median": max(per_rank_medians),
            "min": min(report["timings_ms"]["min"] for report in rank_reports),
            "max": max(report["timings_ms"]["max"] for report in rank_reports),
            "per_rank_median": per_rank_medians,
        },
        "correctness": {
            "max_abs_diff": max_abs_diff,
            "per_rank_max_abs_diff": [report["correctness"]["max_abs_diff"] for report in rank_reports],
        },
        "reference_contract": "host_precomputed_send_order_then_all_to_all_single",
        "rank_reports": rank_reports,
    }
