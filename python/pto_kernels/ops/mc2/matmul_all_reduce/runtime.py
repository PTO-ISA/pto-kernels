"""Runtime helpers for the phase-2 matmul_all_reduce seed."""

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


SUPPORTED_WORLD_SIZES = (2, 4, 8)


@dataclass(frozen=True)
class MatmulAllReduceVariant:
    m: int = 128
    k: int = 256
    n: int = 128
    dtype: str = "float16"
    reduce_op: str = "sum"
    expected_world_size: int = 2
    seed: int = 0
    input_scale: float = 0.125

    def as_dict(self) -> dict[str, int | str | float]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"m{self.m}_k{self.k}_n{self.n}_w{self.expected_world_size}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x1_local": [self.m, self.k],
            "x2": [self.k, self.n],
            "output": [self.m, self.n],
            "world_size": self.expected_world_size,
        }


VARIANT = MatmulAllReduceVariant()
VARIANTS = (
    MatmulAllReduceVariant(m=128, k=256, n=128, expected_world_size=2, seed=0),
    MatmulAllReduceVariant(m=256, k=256, n=128, expected_world_size=2, seed=1),
)


def _resolve_variant(variant: MatmulAllReduceVariant | None = None) -> MatmulAllReduceVariant:
    return VARIANT if variant is None else variant


def _make_rank_tensors(rank: int, variant: MatmulAllReduceVariant | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    resolved = _resolve_variant(variant)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(resolved.seed + rank)
    x1 = (
        torch.randn((resolved.m, resolved.k), generator=generator, dtype=torch.float32) * resolved.input_scale
    ).to(torch.float16)
    generator.manual_seed(resolved.seed + 1000)
    x2 = (
        torch.randn((resolved.k, resolved.n), generator=generator, dtype=torch.float32) * resolved.input_scale
    ).to(torch.float16)
    return x1, x2


def _reference_output(world_size: int, variant: MatmulAllReduceVariant | None = None) -> torch.Tensor:
    resolved = _resolve_variant(variant)
    x2 = _make_rank_tensors(0, resolved)[1]
    full = None
    for rank in range(world_size):
        x1, _ = _make_rank_tensors(rank, resolved)
        local = x1.float() @ x2.float()
        full = local if full is None else full + local
    if full is None:
        raise RuntimeError("Failed to build matmul_all_reduce reference.")
    return full.to(torch.float16)


def _get_hccl_comm_name(rank: int) -> str:
    from torch.distributed.distributed_c10d import _get_default_group

    default_pg = _get_default_group()
    try:
        backend = default_pg._get_backend(torch.device("npu"))
        return backend.get_hccl_comm_name(rank)
    except AttributeError:
        return default_pg.get_hccl_comm_name(rank)


def baseline_blocker(*, device_index: int) -> dict[str, object]:
    symbol_available = hasattr(torch_npu, "npu_mm_all_reduce_base")
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
            "reason": "torch_npu.npu_mm_all_reduce_base is unavailable on this host.",
            "environment": environment,
        }
    if detected_npus < VARIANT.expected_world_size:
        return {
            "status": "blocked",
            "reason": (
                f"Need {VARIANT.expected_world_size} NPUs for matmul_all_reduce bring-up, "
                f"but only {detected_npus} detected."
            ),
            "environment": environment,
        }
    return {
        "status": "ready",
        "environment": environment,
        "entrypoint": "torch_npu.npu_mm_all_reduce_base",
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
    variant = MatmulAllReduceVariant(**variant_dict)
    x1_cpu, x2_cpu = _make_rank_tensors(rank, variant)
    x1 = x1_cpu.to(device)
    x2 = x2_cpu.to(device)
    reference = _reference_output(world_size, variant)
    hcom_name = _get_hccl_comm_name(rank)

    for _ in range(warmup):
        dist.barrier()
        torch_npu.npu_mm_all_reduce_base(x1=x1, x2=x2, hcom=hcom_name, reduce_op=variant.reduce_op)
    torch.npu.synchronize()
    dist.barrier()

    timings_ms: list[float] = []
    output = None
    for _ in range(repeat):
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        output = torch_npu.npu_mm_all_reduce_base(
            x1=x1,
            x2=x2,
            hcom=hcom_name,
            reduce_op=variant.reduce_op,
        )
        torch.npu.synchronize()
        dist.barrier()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    if output is None:
        raise RuntimeError("npu_mm_all_reduce_base did not return an output tensor.")

    max_abs_diff = (output.float().cpu() - reference.float()).abs().max().item()
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


def run_distributed_baseline_benchmark(
    *,
    variant: MatmulAllReduceVariant = VARIANT,
    artifacts_dir: Path,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    world_size = int(os.environ.get("PTO_MC2_MM_AR_WORLD_SIZE", variant.expected_world_size))
    if world_size not in SUPPORTED_WORLD_SIZES:
        return {
            "status": "blocked",
            "reason": f"Unsupported world size {world_size}; expected one of {SUPPORTED_WORLD_SIZES}.",
            "variant": variant.as_dict(),
        }

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
            "reason": launch.get("reason", "Distributed matmul_all_reduce baseline failed."),
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
        "entrypoint": "torch_npu.npu_mm_all_reduce_base",
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
        "reference_contract": "all_reduce(sum_i(x1_local_i @ x2))",
        "rank_reports": rank_reports,
    }


def _load_kernel_module():
    kernel_path = Path(__file__).with_name("kernel.py")
    spec = importlib.util.spec_from_file_location("pto_mc2_matmul_all_reduce_kernel", kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import PTO kernel module from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    variant = MatmulAllReduceVariant(**variant_dict)
    os.environ["PTO_MC2_MM_AR_WORLD_SIZE"] = str(world_size)
    module = _load_kernel_module()
    wrapper = module.build_jit_wrapper(output_dir=output_dir / f"rank_{rank}_kernel")
    build = getattr(wrapper, "_build", None)
    if callable(build):
        build()

    x1_cpu, x2_cpu = _make_rank_tensors(rank, variant)
    x1 = x1_cpu.to(device)
    x2 = x2_cpu.to(device)
    output = torch.empty((variant.m, variant.n), dtype=torch.float16).to(device)
    reference = _reference_output(world_size, variant)

    def _run_once() -> torch.Tensor:
        wrapper(output, x1, x2)
        reduced = output.contiguous()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced

    for _ in range(warmup):
        dist.barrier()
        _run_once()
    torch.npu.synchronize()
    dist.barrier()

    timings_ms: list[float] = []
    pto_output = None
    for _ in range(repeat):
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        pto_output = _run_once()
        torch.npu.synchronize()
        dist.barrier()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    if pto_output is None:
        raise RuntimeError("PTO matmul_all_reduce worker produced no output.")

    max_abs_diff = (pto_output.float().cpu() - reference.float()).abs().max().item()
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
    }


def run_distributed_pto_benchmark(
    *,
    variant: MatmulAllReduceVariant = VARIANT,
    artifacts_dir: Path,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    world_size = int(os.environ.get("PTO_MC2_MM_AR_WORLD_SIZE", variant.expected_world_size))
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
            "reason": launch.get("reason", "Distributed PTO matmul_all_reduce launch failed."),
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
        "reference_contract": "pto_local_matmul_then_all_reduce",
        "rank_reports": rank_reports,
    }
