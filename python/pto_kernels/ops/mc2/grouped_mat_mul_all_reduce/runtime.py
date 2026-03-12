"""Runtime helpers for the phase-2 grouped_mat_mul_all_reduce seed."""

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
class GroupedMatmulAllReduceVariant:
    m: int = 256
    k_total: int = 256
    n: int = 128
    groups: int = 1
    dtype: str = "float16"
    expected_world_size: int = 2
    seed: int = 0
    input_scale: float = 0.125

    def as_dict(self) -> dict[str, int | str | float]:
        return asdict(self)

    @property
    def k_local(self) -> int:
        return self.k_total // self.expected_world_size

    @property
    def label(self) -> str:
        return f"m{self.m}_k{self.k_total}_n{self.n}_w{self.expected_world_size}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x_local": [self.m, self.k_local],
            "weight_local": [self.k_local, self.n],
            "output": [self.m, self.n],
            "group_list": [self.m],
            "world_size": self.expected_world_size,
            "groups": self.groups,
        }


VARIANT = GroupedMatmulAllReduceVariant()
VARIANTS = (
    GroupedMatmulAllReduceVariant(m=128, k_total=256, n=128, expected_world_size=2, seed=0),
    GroupedMatmulAllReduceVariant(m=256, k_total=256, n=128, expected_world_size=2, seed=1),
)


def _resolve_variant(
    variant: GroupedMatmulAllReduceVariant | None = None,
) -> GroupedMatmulAllReduceVariant:
    return VARIANT if variant is None else variant


def _make_rank_tensors(
    rank: int, variant: GroupedMatmulAllReduceVariant | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved = _resolve_variant(variant)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(resolved.seed + rank)
    x = (
        torch.randn((resolved.m, resolved.k_local), generator=generator, dtype=torch.float32)
        * resolved.input_scale
    ).to(torch.float16)
    generator.manual_seed(resolved.seed + 100 + rank)
    weight = (
        torch.randn((resolved.k_local, resolved.n), generator=generator, dtype=torch.float32)
        * resolved.input_scale
    ).to(torch.float16)
    return x, weight


def _grouped_inputs(
    rank: int,
    variant: GroupedMatmulAllReduceVariant | None = None,
    *,
    device: str,
) -> dict[str, object]:
    resolved = _resolve_variant(variant)
    x_cpu, weight_cpu = _make_rank_tensors(rank, resolved)
    x = x_cpu.to(device)
    weight = weight_cpu.to(device)
    weight_v5 = weight.unsqueeze(0).contiguous()
    group_list = torch.tensor([resolved.m], dtype=torch.int64).to(device)
    return {
        "x": x,
        "weight": weight,
        "weight_v5": weight_v5,
        "group_list": group_list,
    }


def _reference_output(
    world_size: int, variant: GroupedMatmulAllReduceVariant | None = None
) -> torch.Tensor:
    resolved = _resolve_variant(variant)
    full = None
    for rank in range(world_size):
        x, weight = _make_rank_tensors(rank, resolved)
        local = x.float() @ weight.float()
        full = local if full is None else full + local
    if full is None:
        raise RuntimeError("Failed to build grouped_mat_mul_all_reduce reference.")
    return full.to(torch.float16)


def run_grouped_matmul_local(inputs: dict[str, object]):
    return torch_npu.npu_grouped_matmul(
        [inputs["x"]],
        [inputs["weight_v5"]],
        group_list=inputs["group_list"],
        split_item=3,
        group_type=0,
        group_list_type=1,
        act_type=0,
    )


def baseline_blocker(*, device_index: int) -> dict[str, object]:
    symbol_available = hasattr(torch_npu, "npu_grouped_matmul")
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
            "reason": "torch_npu.npu_grouped_matmul is unavailable on this host.",
            "environment": environment,
        }
    if detected_npus < VARIANT.expected_world_size:
        return {
            "status": "blocked",
            "reason": (
                f"Need {VARIANT.expected_world_size} NPUs for grouped_mat_mul_all_reduce bring-up, "
                f"but only {detected_npus} detected."
            ),
            "environment": environment,
        }
    return {
        "status": "ready",
        "environment": environment,
        "entrypoint": "torch_npu.npu_grouped_matmul + torch.distributed.all_reduce",
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
    variant = GroupedMatmulAllReduceVariant(**variant_dict)
    inputs = _grouped_inputs(rank, variant, device=device)
    reference = _reference_output(world_size, variant)

    def _run_once() -> torch.Tensor:
        output = run_grouped_matmul_local(inputs)
        local = output[0] if isinstance(output, (list, tuple)) else output
        reduced = local.contiguous()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced

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
        raise RuntimeError("Baseline grouped_mat_mul_all_reduce worker produced no output.")

    max_abs_diff = (output.float().cpu() - reference.float()).abs().max().item()
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


def run_distributed_baseline_benchmark(
    *,
    variant: GroupedMatmulAllReduceVariant = VARIANT,
    artifacts_dir: Path,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    world_size = int(os.environ.get("PTO_MC2_GMM_AR_WORLD_SIZE", variant.expected_world_size))
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
            "reason": launch.get("reason", "Distributed grouped_mat_mul_all_reduce baseline failed."),
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
        "entrypoint": "torch_npu.npu_grouped_matmul + dist.all_reduce",
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
        "reference_contract": "all_reduce(sum_i(grouped_matmul_local_i))",
        "rank_reports": rank_reports,
    }


def _load_kernel_module():
    kernel_path = Path(__file__).with_name("kernel.py")
    spec = importlib.util.spec_from_file_location("pto_mc2_grouped_mat_mul_all_reduce_kernel", kernel_path)
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
    variant = GroupedMatmulAllReduceVariant(**variant_dict)
    os.environ["PTO_MC2_GMM_AR_WORLD_SIZE"] = str(world_size)
    module = _load_kernel_module()
    wrapper = module.build_jit_wrapper(output_dir=output_dir / f"rank_{rank}_kernel")
    build = getattr(wrapper, "_build", None)
    if callable(build):
        build()

    x_cpu, weight_cpu = _make_rank_tensors(rank, variant)
    x = x_cpu.to(device)
    weight = weight_cpu.to(device)
    output = torch.empty((variant.m, variant.n), dtype=torch.float16).to(device)
    reference = _reference_output(world_size, variant)

    def _run_once() -> torch.Tensor:
        wrapper(output, x, weight)
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
        raise RuntimeError("PTO grouped_mat_mul_all_reduce worker produced no output.")

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
    variant: GroupedMatmulAllReduceVariant = VARIANT,
    artifacts_dir: Path,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    world_size = int(os.environ.get("PTO_MC2_GMM_AR_WORLD_SIZE", variant.expected_world_size))
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
            "reason": launch.get("reason", "Distributed PTO grouped_mat_mul_all_reduce launch failed."),
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
        "reference_contract": "pto_local_grouped_matmul_then_all_reduce",
        "rank_reports": rank_reports,
    }
