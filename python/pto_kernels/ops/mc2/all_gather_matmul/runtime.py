"""Runtime helpers for the phase-2 all_gather_matmul seed."""

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
class AllGatherMatmulVariant:
    global_m: int = 128
    k: int = 256
    n: int = 128
    dtype: str = "float16"
    gather_index: int = 0
    gather_output: bool = True
    comm_turn: int = 0
    expected_world_size: int = 2
    seed: int = 0
    input_scale: float = 0.125

    def as_dict(self) -> dict[str, int | str | bool | float]:
        return asdict(self)

    @property
    def local_m(self) -> int:
        return self.global_m // self.expected_world_size

    @property
    def label(self) -> str:
        return f"gm{self.global_m}_k{self.k}_n{self.n}_w{self.expected_world_size}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x1_local": [self.local_m, self.k],
            "x2": [self.k, self.n],
            "gather_out": [self.global_m, self.k],
            "output": [self.global_m, self.n],
            "world_size": self.expected_world_size,
        }


VARIANT = AllGatherMatmulVariant()
VARIANTS = (
    AllGatherMatmulVariant(global_m=128, k=256, n=128, expected_world_size=2, seed=0),
    AllGatherMatmulVariant(global_m=256, k=256, n=128, expected_world_size=2, seed=1),
)


def _resolve_variant(variant: AllGatherMatmulVariant | None = None) -> AllGatherMatmulVariant:
    return VARIANT if variant is None else variant


def _make_local_x1(rank: int, variant: AllGatherMatmulVariant | None = None) -> torch.Tensor:
    resolved = _resolve_variant(variant)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(resolved.seed + rank)
    x1 = (
        torch.randn((resolved.local_m, resolved.k), generator=generator, dtype=torch.float32)
        * resolved.input_scale
    ).to(torch.float16)
    return x1


def _make_x2(variant: AllGatherMatmulVariant | None = None) -> torch.Tensor:
    resolved = _resolve_variant(variant)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(resolved.seed + 1000)
    x2 = (
        torch.randn((resolved.k, resolved.n), generator=generator, dtype=torch.float32) * resolved.input_scale
    ).to(torch.float16)
    return x2


def _reference_outputs(
    world_size: int,
    variant: AllGatherMatmulVariant | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved = _resolve_variant(variant)
    gathered = torch.cat([_make_local_x1(rank, resolved) for rank in range(world_size)], dim=0).contiguous()
    x2 = _make_x2(resolved)
    output = (gathered.float() @ x2.float()).to(torch.float16)
    return output, gathered


def _get_hccl_comm_name(rank: int) -> str:
    from torch.distributed.distributed_c10d import _get_default_group

    default_pg = _get_default_group()
    try:
        backend = default_pg._get_backend(torch.device("npu"))
        return backend.get_hccl_comm_name(rank)
    except AttributeError:
        return default_pg.get_hccl_comm_name(rank)


def baseline_blocker(*, device_index: int) -> dict[str, object]:
    symbol_available = hasattr(torch_npu, "npu_all_gather_base_mm")
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
            "reason": "torch_npu.npu_all_gather_base_mm is unavailable on this host.",
            "environment": environment,
        }
    if detected_npus < VARIANT.expected_world_size:
        return {
            "status": "blocked",
            "reason": (
                f"Need {VARIANT.expected_world_size} NPUs for all_gather_matmul bring-up, "
                f"but only {detected_npus} detected."
            ),
            "environment": environment,
        }
    return {
        "status": "ready",
        "environment": environment,
        "entrypoint": "torch_npu.npu_all_gather_base_mm",
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
    variant = AllGatherMatmulVariant(**variant_dict)
    x1 = _make_local_x1(rank, variant).npu()
    x2 = _make_x2(variant).npu()
    reference_output, reference_gather = _reference_outputs(world_size, variant)
    hcom_name = _get_hccl_comm_name(rank)

    for _ in range(warmup):
        dist.barrier()
        torch_npu.npu_all_gather_base_mm(
            x1,
            x2,
            hcom_name,
            world_size,
            gather_index=variant.gather_index,
            gather_output=variant.gather_output,
            comm_turn=variant.comm_turn,
        )
    torch.npu.synchronize()
    dist.barrier()

    timings_ms: list[float] = []
    output = None
    gather_out = None
    for _ in range(repeat):
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        output, gather_out = torch_npu.npu_all_gather_base_mm(
            x1,
            x2,
            hcom_name,
            world_size,
            gather_index=variant.gather_index,
            gather_output=variant.gather_output,
            comm_turn=variant.comm_turn,
        )
        torch.npu.synchronize()
        dist.barrier()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    if output is None or gather_out is None:
        raise RuntimeError("npu_all_gather_base_mm did not return the expected output tensors.")

    output_diff = (output.float().cpu() - reference_output.float()).abs().max().item()
    gather_diff = (gather_out.float().cpu() - reference_gather.float()).abs().max().item()
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
            "output_max_abs_diff": output_diff,
            "gather_max_abs_diff": gather_diff,
            "max_abs_diff": max(output_diff, gather_diff),
        },
    }


def run_distributed_baseline_benchmark(
    *,
    variant: AllGatherMatmulVariant = VARIANT,
    artifacts_dir: Path,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    world_size = int(os.environ.get("PTO_MC2_ALL_GATHER_WORLD_SIZE", variant.expected_world_size))
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
            "reason": launch.get("reason", "Distributed HCCL baseline bring-up failed."),
            "variant": variant.as_dict(),
            "world_size": world_size,
            "rank_reports": launch.get("rank_reports", []),
        }

    rank_reports = launch["rank_reports"]
    per_rank_medians = [report["timings_ms"]["median"] for report in rank_reports]
    output_max_abs_diff = max(report["correctness"]["output_max_abs_diff"] for report in rank_reports)
    gather_max_abs_diff = max(report["correctness"]["gather_max_abs_diff"] for report in rank_reports)
    max_abs_diff = max(report["correctness"]["max_abs_diff"] for report in rank_reports)
    return {
        "status": "ok",
        "variant": variant.as_dict(),
        "entrypoint": "torch_npu.npu_all_gather_base_mm",
        "world_size": world_size,
        "shape_summary": variant.shape_summary,
        "timings_ms": {
            "median": max(per_rank_medians),
            "min": min(report["timings_ms"]["min"] for report in rank_reports),
            "max": max(report["timings_ms"]["max"] for report in rank_reports),
            "per_rank_median": per_rank_medians,
        },
        "correctness": {
            "output_max_abs_diff": output_max_abs_diff,
            "gather_max_abs_diff": gather_max_abs_diff,
            "max_abs_diff": max_abs_diff,
            "per_rank_max_abs_diff": [report["correctness"]["max_abs_diff"] for report in rank_reports],
        },
        "reference_contract": "output = allgather(x1) @ x2; gather_out = allgather(x1)",
        "rank_reports": rank_reports,
    }


def _load_kernel_module():
    kernel_path = Path(__file__).with_name("kernel.py")
    spec = importlib.util.spec_from_file_location("pto_mc2_all_gather_matmul_kernel", kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import PTO kernel module from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_gather_input(local_x1: torch.Tensor, world_size: int) -> torch.Tensor:
    gathered_parts = [torch.empty_like(local_x1) for _ in range(world_size)]
    dist.all_gather(gathered_parts, local_x1)
    return torch.cat(gathered_parts, dim=0).contiguous()


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
    variant = AllGatherMatmulVariant(**variant_dict)
    os.environ["PTO_MC2_ALL_GATHER_WORLD_SIZE"] = str(world_size)
    os.environ["PTO_MC2_ALL_GATHER_RANK"] = str(rank)
    module = _load_kernel_module()
    wrapper = module.build_jit_wrapper(output_dir=output_dir / f"rank_{rank}_kernel")
    build = getattr(wrapper, "_build", None)
    if callable(build):
        build()

    local_x1 = _make_local_x1(rank, variant).npu()
    x2 = _make_x2(variant).npu()
    reference_output, reference_gather = _reference_outputs(world_size, variant)
    output = torch.empty((variant.global_m, variant.n), dtype=torch.float16).npu()

    def _run_once() -> tuple[torch.Tensor, torch.Tensor]:
        gathered = _all_gather_input(local_x1, world_size)
        wrapper(output, gathered, x2)
        return output, gathered

    for _ in range(warmup):
        dist.barrier()
        _run_once()
    torch.npu.synchronize()
    dist.barrier()

    timings_ms: list[float] = []
    pto_output = None
    pto_gather = None
    for _ in range(repeat):
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        pto_output, pto_gather = _run_once()
        torch.npu.synchronize()
        dist.barrier()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    if pto_output is None or pto_gather is None:
        raise RuntimeError("PTO all_gather_matmul worker did not produce output tensors.")

    output_diff = (pto_output.float().cpu() - reference_output.float()).abs().max().item()
    gather_diff = (pto_gather.float().cpu() - reference_gather.float()).abs().max().item()
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
            "output_max_abs_diff": output_diff,
            "gather_max_abs_diff": gather_diff,
            "max_abs_diff": max(output_diff, gather_diff),
        },
    }


def run_distributed_pto_benchmark(
    *,
    variant: AllGatherMatmulVariant = VARIANT,
    artifacts_dir: Path,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    world_size = int(os.environ.get("PTO_MC2_ALL_GATHER_WORLD_SIZE", variant.expected_world_size))
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
            "reason": launch.get("reason", "Distributed PTO all_gather_matmul launch failed."),
            "variant": variant.as_dict(),
            "world_size": world_size,
            "rank_reports": launch.get("rank_reports", []),
        }

    rank_reports = launch["rank_reports"]
    per_rank_medians = [report["timings_ms"]["median"] for report in rank_reports]
    output_max_abs_diff = max(report["correctness"]["output_max_abs_diff"] for report in rank_reports)
    gather_max_abs_diff = max(report["correctness"]["gather_max_abs_diff"] for report in rank_reports)
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
            "output_max_abs_diff": output_max_abs_diff,
            "gather_max_abs_diff": gather_max_abs_diff,
            "max_abs_diff": max_abs_diff,
            "per_rank_max_abs_diff": [report["correctness"]["max_abs_diff"] for report in rank_reports],
        },
        "reference_contract": "host_all_gather_then_pto_global_matmul",
        "rank_reports": rank_reports,
    }
