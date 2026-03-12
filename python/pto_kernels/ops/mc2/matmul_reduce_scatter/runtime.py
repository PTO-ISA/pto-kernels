"""Runtime helpers for the phase-1 matmul_reduce_scatter seed."""

from __future__ import annotations

import os
import statistics
import time
import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from pto_kernels.utils import run_local_ranked_job


SUPPORTED_WORLD_SIZES = (2, 4, 8)


@dataclass(frozen=True)
class MatmulReduceScatterVariant:
    m: int = 128
    k: int = 256
    n: int = 128
    dtype: str = "float16"
    reduce_op: str = "sum"
    bias: bool = False
    comm_turn: int = 0
    stream_mode: int = 1
    expected_world_size: int = 2
    seed: int = 0
    input_scale: float = 0.125

    def as_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)


VARIANT = MatmulReduceScatterVariant()

def resolve_world_size() -> int:
    value = os.environ.get("PTO_MC2_WORLD_SIZE")
    if value is None:
        return VARIANT.expected_world_size
    return int(value)


def _make_rank_tensors(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(VARIANT.seed + rank)
    x1 = (
        torch.randn((VARIANT.m, VARIANT.k), generator=generator, dtype=torch.float32) * VARIANT.input_scale
    ).to(torch.float16)
    generator.manual_seed(VARIANT.seed + 100 + rank)
    x2 = (
        torch.randn((VARIANT.k, VARIANT.n), generator=generator, dtype=torch.float32) * VARIANT.input_scale
    ).to(torch.float16)
    return x1, x2


def _reference_chunk(rank: int, world_size: int) -> torch.Tensor:
    full = None
    for peer_rank in range(world_size):
        x1, x2 = _make_rank_tensors(peer_rank)
        mm = x1.float() @ x2.float()
        full = mm if full is None else full + mm
    if full is None:
        raise RuntimeError("Failed to build reference matmul for reduce_scatter.")
    return full.chunk(world_size, dim=0)[rank].contiguous()


def _get_hccl_comm_name(rank: int) -> str:
    from torch.distributed.distributed_c10d import _get_default_group

    default_pg = _get_default_group()
    try:
        backend = default_pg._get_backend(torch.device("npu"))
        return backend.get_hccl_comm_name(rank)
    except AttributeError:
        return default_pg.get_hccl_comm_name(rank)


def _baseline_worker(*, rank: int, world_size: int, output_dir: Path, device: str, warmup: int, repeat: int):
    del output_dir
    x1_cpu, x2_cpu = _make_rank_tensors(rank)
    x1 = x1_cpu.npu()
    x2 = x2_cpu.npu()
    reference = _reference_chunk(rank, world_size)
    hcom_name = _get_hccl_comm_name(rank)

    for _ in range(warmup):
        dist.barrier()
        torch_npu.npu_mm_reduce_scatter_base(
            x1, x2, hcom_name, world_size, reduce_op=VARIANT.reduce_op
        )
    torch.npu.synchronize()
    dist.barrier()

    timings_ms: list[float] = []
    output = None
    for _ in range(repeat):
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        output = torch_npu.npu_mm_reduce_scatter_base(
            x1, x2, hcom_name, world_size, reduce_op=VARIANT.reduce_op
        )
        torch.npu.synchronize()
        dist.barrier()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    if output is None:
        raise RuntimeError("npu_mm_reduce_scatter_base did not return an output tensor.")

    max_abs_diff = (output.float().cpu() - reference).abs().max().item()
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
        "correctness": {"max_abs_diff": max_abs_diff},
    }


def run_distributed_baseline_benchmark(*, artifacts_dir: Path, warmup: int, repeat: int) -> dict[str, object]:
    world_size = resolve_world_size()
    if world_size not in SUPPORTED_WORLD_SIZES:
        return {
            "status": "blocked",
            "reason": (
                f"PTO_MC2_WORLD_SIZE={world_size} is unsupported; expected one of {SUPPORTED_WORLD_SIZES}."
            ),
            "variant": VARIANT.as_dict(),
        }

    output_dir = Path(artifacts_dir) / "distributed_baseline"
    launch = run_local_ranked_job(
        _baseline_worker,
        world_size=world_size,
        output_dir=output_dir,
        worker_kwargs={"warmup": warmup, "repeat": repeat},
    )
    if launch["status"] != "ok":
        return {
            "status": "blocked",
            "reason": launch.get("reason", "Distributed HCCL baseline bring-up failed."),
            "variant": VARIANT.as_dict(),
            "world_size": world_size,
            "rank_reports": launch.get("rank_reports", []),
        }

    rank_reports = launch["rank_reports"]
    per_rank_medians = [report["timings_ms"]["median"] for report in rank_reports]
    max_abs_diff = max(report["correctness"]["max_abs_diff"] for report in rank_reports)
    return {
        "status": "ok",
        "variant": VARIANT.as_dict(),
        "entrypoint": "torch_npu.npu_mm_reduce_scatter_base",
        "world_size": world_size,
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
        "reference_contract": "reduce_scatter(sum_i(x1_i @ x2_i))",
        "rank_reports": rank_reports,
    }


def _load_kernel_module():
    kernel_path = Path(__file__).with_name("kernel.py")
    spec = importlib.util.spec_from_file_location("pto_mc2_matmul_reduce_scatter_kernel", kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import PTO kernel module from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pto_worker(*, rank: int, world_size: int, output_dir: Path, device: str, warmup: int, repeat: int):
    module = _load_kernel_module()
    wrapper = module.build_jit_wrapper(output_dir=output_dir / f"rank_{rank}_kernel")
    build = getattr(wrapper, "_build", None)
    if callable(build):
        build()

    x1_cpu, x2_cpu = _make_rank_tensors(rank)
    x1 = x1_cpu.npu()
    x2 = x2_cpu.npu()
    local_mm = torch.empty((VARIANT.m, VARIANT.n), dtype=torch.float16).npu()
    reference = _reference_chunk(rank, world_size)

    def _run_once() -> torch.Tensor:
        wrapper(local_mm, x1, x2)
        dist.all_reduce(local_mm, op=dist.ReduceOp.SUM)
        return local_mm.chunk(world_size, dim=0)[rank].contiguous()

    for _ in range(warmup):
        dist.barrier()
        _run_once()
    torch.npu.synchronize()
    dist.barrier()

    timings_ms: list[float] = []
    output = None
    for _ in range(repeat):
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        output = _run_once()
        torch.npu.synchronize()
        dist.barrier()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    if output is None:
        raise RuntimeError("PTO distributed MC2 seed did not produce an output tensor.")

    max_abs_diff = (output.float().cpu() - reference).abs().max().item()
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
        "correctness": {"max_abs_diff": max_abs_diff},
        "reference_contract": "host_orchestrated_all_reduce_then_chunk",
    }


def run_distributed_pto_benchmark(*, artifacts_dir: Path, warmup: int, repeat: int) -> dict[str, object]:
    world_size = resolve_world_size()
    if world_size not in SUPPORTED_WORLD_SIZES:
        return {
            "status": "blocked",
            "reason": (
                f"PTO_MC2_WORLD_SIZE={world_size} is unsupported; expected one of {SUPPORTED_WORLD_SIZES}."
            ),
            "variant": VARIANT.as_dict(),
        }

    output_dir = Path(artifacts_dir) / "distributed_pto"
    launch = run_local_ranked_job(
        _pto_worker,
        world_size=world_size,
        output_dir=output_dir,
        worker_kwargs={"warmup": warmup, "repeat": repeat},
    )
    if launch["status"] != "ok":
        return {
            "status": "blocked",
            "reason": launch.get("reason", "Distributed PTO MC2 bring-up failed."),
            "variant": VARIANT.as_dict(),
            "world_size": world_size,
            "rank_reports": launch.get("rank_reports", []),
        }

    rank_reports = launch["rank_reports"]
    per_rank_medians = [report["timings_ms"]["median"] for report in rank_reports]
    max_abs_diff = max(report["correctness"]["max_abs_diff"] for report in rank_reports)
    return {
        "status": "ok",
        "variant": VARIANT.as_dict(),
        "entrypoint": "pto_seed_local_mm_plus_hccl_all_reduce_chunk",
        "world_size": world_size,
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
        "reference_contract": "host_orchestrated_all_reduce_then_chunk",
        "rank_reports": rank_reports,
    }


def baseline_environment_summary(*, device_index: int = 0) -> dict[str, object]:
    world_size_env = os.environ.get("WORLD_SIZE")
    try:
        parsed_world_size = int(world_size_env) if world_size_env is not None else None
    except ValueError:
        parsed_world_size = None

    return {
        "device_index": device_index,
        "runtime_entrypoint": "torch_npu.npu_mm_reduce_scatter_base",
        "symbol_available": hasattr(torch_npu, "npu_mm_reduce_scatter_base"),
        "distributed_available": bool(dist.is_available()),
        "distributed_initialized": bool(dist.is_initialized()),
        "dist_backend": dist.get_backend() if dist.is_initialized() else None,
        "world_size_env": world_size_env,
        "parsed_world_size": parsed_world_size,
        "supported_world_sizes": list(SUPPORTED_WORLD_SIZES),
        "resolved_world_size": resolve_world_size(),
        "variant": VARIANT.as_dict(),
    }


def baseline_blocker(*, device_index: int = 0) -> dict[str, object]:
    summary = baseline_environment_summary(device_index=device_index)
    if not summary["symbol_available"]:
        reason = "torch_npu does not expose npu_mm_reduce_scatter_base on this environment."
    elif summary["parsed_world_size"] is None:
        reason = (
            "torch_npu.npu_mm_reduce_scatter_base requires a multi-rank HCCL process group and an hcom "
            "handle from get_hccl_comm_name, but this benchmark is running outside a distributed launcher. "
            "Set up a torch.distributed hccl job with world_size in {2, 4, 8}."
        )
    elif summary["parsed_world_size"] not in SUPPORTED_WORLD_SIZES:
        reason = (
            f"torch_npu.npu_mm_reduce_scatter_base only supports world_size in {SUPPORTED_WORLD_SIZES}, "
            f"but WORLD_SIZE={summary['parsed_world_size']}."
        )
    elif not summary["distributed_initialized"]:
        reason = (
            "WORLD_SIZE is set, but torch.distributed is not initialized with backend='hccl', so the "
            "baseline path cannot obtain an hcom handle for reduce_scatter."
        )
    elif summary["dist_backend"] != "hccl":
        reason = (
            f"torch.distributed is initialized with backend={summary['dist_backend']!r}; "
            "matmul_reduce_scatter requires backend='hccl'."
        )
    else:
        reason = (
            "The current benchmark runner is single-process and does not coordinate the peer ranks required "
            "to execute MC2 reduce_scatter. Run this seed under a dedicated multi-rank hccl harness."
        )
    return {"status": "blocked", "reason": reason, "environment": summary}
