"""Runtime helpers for the phase-2 moe_distribute_combine seed."""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from pto_kernels.ops.mc2.moe_distribute_dispatch.runtime import (
    SUPPORTED_WORLD_SIZES,
    launcher_blocker,
)
from pto_kernels.utils import run_local_ranked_job


@dataclass(frozen=True)
class MoeDistributeCombineVariant:
    tokens: int = 8
    hidden_size: int = 7168
    dtype: str = "float16"
    expected_world_size: int = 8
    topk: int = 1
    seed: int = 0
    input_scale: float = 0.125

    def as_dict(self) -> dict[str, int | str | float]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"t{self.tokens}_h{self.hidden_size}_w{self.expected_world_size}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "compact_expand_x": [self.tokens, self.hidden_size],
            "scatter_indices": [self.tokens, self.hidden_size],
            "x_out": [self.tokens, self.hidden_size],
            "expert_ids_local": [self.tokens, self.topk],
            "expand_idx": [self.tokens * self.topk],
            "ep_send_counts_prefix": [self.expected_world_size],
            "expert_scales": [self.tokens, self.topk],
            "world_size": self.expected_world_size,
        }


VARIANT = MoeDistributeCombineVariant(expected_world_size=8)
VARIANTS = (VARIANT,)


def _write_worker_trace(output_dir: Path, rank: int, stage: str, **extra: object) -> None:
    trace_path = output_dir / f"rank_{rank}.worker_trace.json"
    payload = {"rank": rank, "stage": stage, "timestamp": time.time()}
    payload.update(extra)
    trace_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _device_for(device_index: int) -> str:
    return f"npu:{device_index}"


def _make_rank_tensors(
    rank: int,
    variant: MoeDistributeCombineVariant | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved = VARIANT if variant is None else variant
    generator = torch.Generator(device="cpu")
    generator.manual_seed(resolved.seed + rank)
    x = (
        torch.randn((resolved.tokens, resolved.hidden_size), generator=generator, dtype=torch.float32)
        * resolved.input_scale
    ).to(torch.float16)
    base = torch.arange(resolved.tokens, dtype=torch.int64)
    route = ((base + rank + resolved.seed) % resolved.expected_world_size).to(torch.int32)
    expert_ids = route.unsqueeze(1).contiguous()
    return x, expert_ids


def _block_dim() -> int:
    value = int(os.environ.get("PTO_MC2_MOE_COMBINE_BLOCK_DIM", "4"))
    if value not in (2, 4, 8):
        raise ValueError(f"Unsupported PTO_MC2_MOE_COMBINE_BLOCK_DIM={value}; expected one of 2,4,8.")
    return value


def _expand_idx_for_local_tokens(expert_ids: torch.Tensor) -> torch.Tensor:
    counts: dict[int, int] = {}
    values = []
    for expert in expert_ids.reshape(-1).tolist():
        idx = counts.get(int(expert), 0)
        values.append(idx)
        counts[int(expert)] = idx + 1
    return torch.tensor(values, dtype=torch.int32)


def _ep_send_counts_prefix(expert_ids: torch.Tensor, world_size: int) -> torch.Tensor:
    flat = expert_ids.reshape(-1).to(torch.int64)
    return torch.bincount(flat, minlength=world_size).to(torch.int32)


def make_combine_inputs(
    rank: int,
    variant: MoeDistributeCombineVariant = VARIANT,
    *,
    device_index: int = 0,
) -> dict[str, object]:
    device = _device_for(device_index)
    torch.npu.set_device(device)
    x_cpu, expert_ids_cpu = _make_rank_tensors(rank, variant)
    expand_idx_cpu = _expand_idx_for_local_tokens(expert_ids_cpu)
    ep_send_counts_cpu = _ep_send_counts_prefix(expert_ids_cpu, variant.expected_world_size)
    expert_scales_cpu = torch.ones((variant.tokens, variant.topk), dtype=torch.float32)
    rows_per_core = variant.tokens // _block_dim()
    permutation: list[int] = []
    scatter_rows = torch.empty((variant.tokens, variant.hidden_size), dtype=torch.int16)
    for row_start in range(0, variant.tokens, rows_per_core):
        chunk = list(range(row_start, row_start + rows_per_core))
        rotated = chunk[1:] + chunk[:1] if len(chunk) > 1 else chunk
        for compact_row_offset, original_row in enumerate(rotated):
            compact_row = row_start + compact_row_offset
            permutation.append(original_row)
            base = (original_row - row_start) * variant.hidden_size
            scatter_rows[compact_row] = torch.arange(
                base,
                base + variant.hidden_size,
                dtype=torch.int16,
            )

    compact_expand_x_cpu = x_cpu.index_select(0, torch.tensor(permutation, dtype=torch.int64))
    scatter_indices_cpu = scatter_rows.contiguous()

    return {
        "device": device,
        "variant": variant.as_dict(),
        "shape_summary": variant.shape_summary,
        "compact_expand_x": compact_expand_x_cpu.npu(),
        "scatter_indices": scatter_indices_cpu.npu(),
        "expert_ids": expert_ids_cpu.npu(),
        "expand_idx": expand_idx_cpu.npu(),
        "ep_send_counts_prefix": ep_send_counts_cpu.npu(),
        "expert_scales": expert_scales_cpu.npu(),
        "x_out_pto": torch.zeros_like(x_cpu).npu(),
        "reference_x": x_cpu.float(),
    }


def baseline_blocker(*, device_index: int) -> dict[str, object]:
    symbol_available = hasattr(torch_npu, "npu_moe_distribute_combine")
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
            "reason": "torch_npu.npu_moe_distribute_combine is unavailable on this host.",
            "environment": environment,
        }
    if detected_npus < VARIANT.expected_world_size:
        return {
            "status": "blocked",
            "reason": (
                f"Need {VARIANT.expected_world_size} NPUs for moe_distribute_combine bring-up, "
                f"but only {detected_npus} detected."
            ),
            "environment": environment,
        }
    blocker = launcher_blocker(VARIANT.expected_world_size)
    if blocker is not None:
        blocker["environment"] = environment
        return blocker
    return {
        "status": "ready",
        "environment": environment,
        "entrypoint": "torch_npu.npu_moe_distribute_combine",
    }


def _get_hccl_comm_name(rank: int) -> str:
    from torch.distributed.distributed_c10d import _get_default_group

    default_pg = _get_default_group()
    try:
        backend = default_pg._get_backend(torch.device("npu"))
        return backend.get_hccl_comm_name(rank)
    except AttributeError:
        return default_pg.get_hccl_comm_name(rank)


def _load_kernel_module():
    kernel_path = Path(__file__).with_name("kernel.py")
    spec = importlib.util.spec_from_file_location("pto_mc2_moe_distribute_combine_kernel", kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import PTO kernel module from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _correctness_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    max_abs_diff = (actual.float().cpu() - reference.float().cpu()).abs().max().item()
    return {"x_out_max_abs_diff": max_abs_diff, "max_abs_diff": max_abs_diff}


def _call_baseline_combine(
    compact_expand_x: torch.Tensor,
    expert_ids: torch.Tensor,
    expand_idx: torch.Tensor,
    ep_send_counts_prefix: torch.Tensor,
    expert_scales: torch.Tensor,
    *,
    hcom_name: str,
    world_size: int,
    rank: int,
    variant: MoeDistributeCombineVariant,
) -> torch.Tensor:
    return torch_npu.npu_moe_distribute_combine(
        expand_x=compact_expand_x,
        expert_ids=expert_ids,
        expand_idx=expand_idx,
        ep_send_counts=ep_send_counts_prefix,
        expert_scales=expert_scales,
        group_ep=hcom_name,
        ep_world_size=world_size,
        ep_rank_id=rank,
        moe_expert_num=world_size,
        tp_send_counts=None,
        x_active_mask=None,
        activation_scale=None,
        weight_scale=None,
        group_list=None,
        expand_scales=None,
        shared_expert_x=None,
        group_tp="",
        tp_world_size=0,
        tp_rank_id=0,
        expert_shard_type=0,
        shared_expert_num=0,
        shared_expert_rank_num=0,
        global_bs=variant.tokens * world_size,
        out_dtype=0,
        comm_quant_mode=0,
        group_list_type=0,
    )


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
    variant = MoeDistributeCombineVariant(**variant_dict)
    _write_worker_trace(output_dir, rank, "baseline_inputs")
    inputs = make_combine_inputs(rank, variant, device_index=rank)
    compact_expand_x = inputs["compact_expand_x"]
    expert_ids = inputs["expert_ids"]
    expand_idx = inputs["expand_idx"]
    ep_send_counts_prefix = inputs["ep_send_counts_prefix"]
    expert_scales = inputs["expert_scales"]
    reference_x = inputs["reference_x"]
    hcom_name = _get_hccl_comm_name(rank)

    for iteration in range(warmup):
        _write_worker_trace(output_dir, rank, "baseline_warmup_before_call", iteration=iteration)
        dist.barrier()
        _call_baseline_combine(
            compact_expand_x,
            expert_ids,
            expand_idx,
            ep_send_counts_prefix,
            expert_scales,
            hcom_name=hcom_name,
            world_size=world_size,
            rank=rank,
            variant=variant,
        )
        _write_worker_trace(output_dir, rank, "baseline_warmup_after_call", iteration=iteration)
    torch.npu.synchronize()
    dist.barrier()

    timings_ms: list[float] = []
    actual = None
    for iteration in range(repeat):
        _write_worker_trace(output_dir, rank, "baseline_repeat_before_call", iteration=iteration)
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        actual = _call_baseline_combine(
            compact_expand_x,
            expert_ids,
            expand_idx,
            ep_send_counts_prefix,
            expert_scales,
            hcom_name=hcom_name,
            world_size=world_size,
            rank=rank,
            variant=variant,
        )
        _write_worker_trace(output_dir, rank, "baseline_repeat_after_call", iteration=iteration)
        torch.npu.synchronize()
        dist.barrier()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    if actual is None:
        raise RuntimeError("npu_moe_distribute_combine did not return an output tensor.")

    metrics = _correctness_metrics(actual, reference_x)
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
        "correctness": metrics,
    }


def run_distributed_baseline_benchmark(
    *,
    variant: MoeDistributeCombineVariant = VARIANT,
    artifacts_dir: Path,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    world_size = int(os.environ.get("PTO_MC2_MOE_COMBINE_WORLD_SIZE", variant.expected_world_size))
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
        timeout_seconds=300,
    )
    if launch["status"] != "ok":
        return {
            "status": "blocked",
            "reason": launch.get("reason", "Distributed moe_distribute_combine baseline failed."),
            "variant": variant.as_dict(),
            "world_size": world_size,
            "rank_reports": launch.get("rank_reports", []),
            "rank_stages": launch.get("rank_stages", []),
        }

    rank_reports = launch["rank_reports"]
    per_rank_medians = [report["timings_ms"]["median"] for report in rank_reports]
    max_abs_diff = max(report["correctness"]["max_abs_diff"] for report in rank_reports)
    return {
        "status": "ok",
        "variant": variant.as_dict(),
        "entrypoint": "torch_npu.npu_moe_distribute_combine",
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
        "reference_contract": "host_precompacted_reverse_route_buffer",
        "rank_reports": rank_reports,
    }


def run_pto_combine_variant(wrapper, inputs: dict[str, object]):
    wrapper(inputs["x_out_pto"], inputs["compact_expand_x"], inputs["scatter_indices"])
    return inputs["x_out_pto"].float()


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
    del device
    variant = MoeDistributeCombineVariant(**variant_dict)
    os.environ["PTO_MC2_MOE_COMBINE_WORLD_SIZE"] = str(world_size)
    module = _load_kernel_module()
    wrapper = module.build_jit_wrapper(output_dir=output_dir / f"rank_{rank}_kernel")
    build = getattr(wrapper, "_build", None)
    if callable(build):
        build()

    inputs = make_combine_inputs(rank, variant, device_index=rank)
    reference_x = inputs["reference_x"]
    _write_worker_trace(output_dir, rank, "pto_inputs")

    for iteration in range(warmup):
        _write_worker_trace(output_dir, rank, "pto_warmup_before_call", iteration=iteration)
        dist.barrier()
        run_pto_combine_variant(wrapper, inputs)
        _write_worker_trace(output_dir, rank, "pto_warmup_after_call", iteration=iteration)
    torch.npu.synchronize()
    dist.barrier()

    timings_ms: list[float] = []
    actual = None
    for iteration in range(repeat):
        _write_worker_trace(output_dir, rank, "pto_repeat_before_call", iteration=iteration)
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        actual = run_pto_combine_variant(wrapper, inputs)
        _write_worker_trace(output_dir, rank, "pto_repeat_after_call", iteration=iteration)
        torch.npu.synchronize()
        dist.barrier()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    if actual is None:
        raise RuntimeError("PTO moe_distribute_combine worker produced no output.")

    metrics = _correctness_metrics(actual, reference_x)
    return {
        "status": "ok",
        "rank": rank,
        "world_size": world_size,
        "timings_ms": {
            "median": statistics.median(timings_ms),
            "min": min(timings_ms),
            "max": max(timings_ms),
        },
        "correctness": metrics,
        "reference_contract": "host_precompacted_reverse_route_buffer",
    }


def run_distributed_pto_benchmark(
    *,
    variant: MoeDistributeCombineVariant = VARIANT,
    artifacts_dir: Path,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    world_size = int(os.environ.get("PTO_MC2_MOE_COMBINE_WORLD_SIZE", variant.expected_world_size))
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
        timeout_seconds=300,
    )
    if launch["status"] != "ok":
        return {
            "status": "blocked",
            "reason": launch.get("reason", "Distributed PTO moe_distribute_combine failed."),
            "variant": variant.as_dict(),
            "world_size": world_size,
            "rank_reports": launch.get("rank_reports", []),
            "rank_stages": launch.get("rank_stages", []),
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
        "reference_contract": "host_precompacted_reverse_route_buffer",
        "rank_reports": rank_reports,
    }
