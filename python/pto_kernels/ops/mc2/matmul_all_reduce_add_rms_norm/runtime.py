"""Runtime helpers for the phase-2 matmul_all_reduce_add_rms_norm seed."""

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
class MatmulAllReduceAddRmsNormVariant:
    m: int = 128
    k: int = 256
    n: int = 128
    dtype: str = "float16"
    reduce_op: str = "sum"
    epsilon: float = 1e-3
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
            "residual": [self.m, self.n],
            "gamma": [self.n],
            "y": [self.m, self.n],
            "norm_out": [self.m, self.n],
            "world_size": self.expected_world_size,
        }


VARIANT = MatmulAllReduceAddRmsNormVariant()
VARIANTS = (
    MatmulAllReduceAddRmsNormVariant(m=128, k=256, n=128, expected_world_size=2, seed=0),
    MatmulAllReduceAddRmsNormVariant(m=256, k=256, n=128, expected_world_size=2, seed=1),
)


def _resolve_variant(
    variant: MatmulAllReduceAddRmsNormVariant | None = None,
) -> MatmulAllReduceAddRmsNormVariant:
    return VARIANT if variant is None else variant


def _make_rank_tensors(
    rank: int,
    variant: MatmulAllReduceAddRmsNormVariant | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    generator.manual_seed(resolved.seed + 200 + rank)
    residual = (
        torch.randn((resolved.m, resolved.n), generator=generator, dtype=torch.float32) * resolved.input_scale
    ).to(torch.float16)
    generator.manual_seed(resolved.seed + 3000)
    gamma = (
        torch.randn((resolved.n,), generator=generator, dtype=torch.float32) * resolved.input_scale + 1.0
    ).to(torch.float16)
    return x1, x2, residual, gamma


def _reference_outputs(
    rank: int,
    world_size: int,
    variant: MatmulAllReduceAddRmsNormVariant | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved = _resolve_variant(variant)
    x2 = _make_rank_tensors(0, resolved)[1]
    residual = _make_rank_tensors(rank, resolved)[2]
    gamma = _make_rank_tensors(0, resolved)[3]
    mm_full = None
    for peer_rank in range(world_size):
        x1, _, _, _ = _make_rank_tensors(peer_rank, resolved)
        local = x1.float() @ x2.float()
        mm_full = local if mm_full is None else mm_full + local
    if mm_full is None:
        raise RuntimeError("Failed to build matmul_all_reduce_add_rms_norm reference.")
    y = mm_full + residual.float()
    rstd = torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + resolved.epsilon)
    norm_out = (y * rstd * gamma.float()).to(torch.float16)
    return y.to(torch.float16), norm_out


def _get_hccl_comm_name(rank: int) -> str:
    from torch.distributed.distributed_c10d import _get_default_group

    default_pg = _get_default_group()
    try:
        backend = default_pg._get_backend(torch.device("npu"))
        return backend.get_hccl_comm_name(rank)
    except AttributeError:
        return default_pg.get_hccl_comm_name(rank)


def baseline_blocker(*, device_index: int) -> dict[str, object]:
    mm_available = hasattr(torch_npu, "npu_mm_all_reduce_base")
    arn_available = hasattr(torch_npu, "npu_add_rms_norm")
    npu_available = torch.npu.is_available()
    detected_npus = torch.npu.device_count() if npu_available else 0
    environment = {
        "device_index": device_index,
        "npu_available": bool(npu_available),
        "detected_npus": int(detected_npus),
        "mm_symbol_available": bool(mm_available),
        "arn_symbol_available": bool(arn_available),
    }
    if not mm_available:
        return {
            "status": "blocked",
            "reason": "torch_npu.npu_mm_all_reduce_base is unavailable on this host.",
            "environment": environment,
        }
    if not arn_available:
        return {
            "status": "blocked",
            "reason": "torch_npu.npu_add_rms_norm is unavailable on this host.",
            "environment": environment,
        }
    if detected_npus < VARIANT.expected_world_size:
        return {
            "status": "blocked",
            "reason": (
                f"Need {VARIANT.expected_world_size} NPUs for matmul_all_reduce_add_rms_norm bring-up, "
                f"but only {detected_npus} detected."
            ),
            "environment": environment,
        }
    return {
        "status": "ready",
        "environment": environment,
        "entrypoint": "torch_npu.npu_mm_all_reduce_base + torch_npu.npu_add_rms_norm",
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
    variant = MatmulAllReduceAddRmsNormVariant(**variant_dict)
    x1_cpu, x2_cpu, residual_cpu, gamma_cpu = _make_rank_tensors(rank, variant)
    x1 = x1_cpu.to(device)
    x2 = x2_cpu.to(device)
    residual = residual_cpu.to(device)
    gamma = gamma_cpu.to(device)
    ref_y, ref_norm = _reference_outputs(rank, world_size, variant)
    hcom_name = _get_hccl_comm_name(rank)

    for _ in range(warmup):
        dist.barrier()
        mm_out = torch_npu.npu_mm_all_reduce_base(x1=x1, x2=x2, hcom=hcom_name, reduce_op=variant.reduce_op)
        torch_npu.npu_add_rms_norm(mm_out, residual, gamma, variant.epsilon)
    torch.npu.synchronize()
    dist.barrier()

    timings_ms: list[float] = []
    norm_out = None
    y_out = None
    for _ in range(repeat):
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        mm_out = torch_npu.npu_mm_all_reduce_base(x1=x1, x2=x2, hcom=hcom_name, reduce_op=variant.reduce_op)
        norm_out, _, y_out = torch_npu.npu_add_rms_norm(mm_out, residual, gamma, variant.epsilon)
        torch.npu.synchronize()
        dist.barrier()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    if norm_out is None or y_out is None:
        raise RuntimeError("Baseline matmul_all_reduce_add_rms_norm worker produced no output.")

    y_diff = (y_out.float().cpu() - ref_y.float()).abs().max().item()
    norm_diff = (norm_out.float().cpu() - ref_norm.float()).abs().max().item()
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
            "y_max_abs_diff": y_diff,
            "norm_max_abs_diff": norm_diff,
            "max_abs_diff": max(y_diff, norm_diff),
        },
    }


def run_distributed_baseline_benchmark(
    *,
    variant: MatmulAllReduceAddRmsNormVariant = VARIANT,
    artifacts_dir: Path,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    world_size = int(os.environ.get("PTO_MC2_MM_ARN_WORLD_SIZE", variant.expected_world_size))
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
            "reason": launch.get("reason", "Distributed matmul_all_reduce_add_rms_norm baseline failed."),
            "variant": variant.as_dict(),
            "world_size": world_size,
            "rank_reports": launch.get("rank_reports", []),
        }

    rank_reports = launch["rank_reports"]
    per_rank_medians = [report["timings_ms"]["median"] for report in rank_reports]
    y_diff = max(report["correctness"]["y_max_abs_diff"] for report in rank_reports)
    norm_diff = max(report["correctness"]["norm_max_abs_diff"] for report in rank_reports)
    max_abs_diff = max(report["correctness"]["max_abs_diff"] for report in rank_reports)
    return {
        "status": "ok",
        "variant": variant.as_dict(),
        "entrypoint": "torch_npu.npu_mm_all_reduce_base + torch_npu.npu_add_rms_norm",
        "world_size": world_size,
        "shape_summary": variant.shape_summary,
        "timings_ms": {
            "median": max(per_rank_medians),
            "min": min(report["timings_ms"]["min"] for report in rank_reports),
            "max": max(report["timings_ms"]["max"] for report in rank_reports),
            "per_rank_median": per_rank_medians,
        },
        "correctness": {
            "y_max_abs_diff": y_diff,
            "norm_max_abs_diff": norm_diff,
            "max_abs_diff": max_abs_diff,
            "per_rank_max_abs_diff": [report["correctness"]["max_abs_diff"] for report in rank_reports],
        },
        "reference_contract": "add_rms_norm(all_reduce(sum_i(x1_local_i @ x2)), residual_rank, gamma)",
        "rank_reports": rank_reports,
    }


def _load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import PTO kernel module from {path}")
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
    variant = MatmulAllReduceAddRmsNormVariant(**variant_dict)
    os.environ["PTO_MC2_MM_AR_WORLD_SIZE"] = str(world_size)
    os.environ["PTO_MC2_MM_AR_M"] = str(variant.m)
    os.environ["PTO_MC2_MM_AR_K"] = str(variant.k)
    os.environ["PTO_MC2_MM_AR_N"] = str(variant.n)
    os.environ["PTO_MC2_MM_ARN_WORLD_SIZE"] = str(world_size)
    os.environ["PTO_MC2_MM_ARN_M"] = str(variant.m)
    os.environ["PTO_MC2_MM_ARN_N"] = str(variant.n)

    mm_module = _load_module_from_path(
        Path(__file__).resolve().parent.parent / "matmul_all_reduce" / "kernel.py",
        "pto_mc2_matmul_all_reduce_kernel_for_arn",
    )
    arn_module = _load_module_from_path(
        Path(__file__).with_name("kernel.py"),
        "pto_mc2_matmul_all_reduce_add_rms_norm_kernel",
    )

    mm_wrapper = mm_module.build_jit_wrapper(output_dir=output_dir / f"rank_{rank}_mm_kernel")
    arn_wrapper = arn_module.build_add_rms_norm_jit_wrapper(output_dir=output_dir / f"rank_{rank}_arn_kernel")
    for wrapper in (mm_wrapper, arn_wrapper):
        build = getattr(wrapper, "_build", None)
        if callable(build):
            build()

    x1_cpu, x2_cpu, residual_cpu, gamma_cpu = _make_rank_tensors(rank, variant)
    x1 = x1_cpu.to(device)
    x2 = x2_cpu.to(device)
    residual = residual_cpu.to(device)
    gamma = gamma_cpu.to(device)
    inv_n = torch.full((1, variant.n), 1.0 / float(variant.n), dtype=torch.float16).to(device)
    eps_row = torch.full((1, variant.n), float(variant.epsilon), dtype=torch.float16).to(device)
    mm_out = torch.empty((variant.m, variant.n), dtype=torch.float16).to(device)
    y_out = torch.empty((variant.m, variant.n), dtype=torch.float16).to(device)
    norm_out = torch.empty((variant.m, variant.n), dtype=torch.float16).to(device)
    ref_y, ref_norm = _reference_outputs(rank, world_size, variant)

    def _run_once() -> tuple[torch.Tensor, torch.Tensor]:
        mm_wrapper(mm_out, x1, x2)
        dist.all_reduce(mm_out, op=dist.ReduceOp.SUM)
        arn_wrapper(y_out, norm_out, mm_out, residual, gamma.view(1, variant.n), inv_n, eps_row)
        return y_out, norm_out

    for _ in range(warmup):
        dist.barrier()
        _run_once()
    torch.npu.synchronize()
    dist.barrier()

    timings_ms: list[float] = []
    pto_y = None
    pto_norm = None
    for _ in range(repeat):
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        pto_y, pto_norm = _run_once()
        torch.npu.synchronize()
        dist.barrier()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    if pto_y is None or pto_norm is None:
        raise RuntimeError("PTO matmul_all_reduce_add_rms_norm worker produced no output.")

    y_diff = (pto_y.float().cpu() - ref_y.float()).abs().max().item()
    norm_diff = (pto_norm.float().cpu() - ref_norm.float()).abs().max().item()
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
            "y_max_abs_diff": y_diff,
            "norm_max_abs_diff": norm_diff,
            "max_abs_diff": max(y_diff, norm_diff),
        },
    }


def run_distributed_pto_benchmark(
    *,
    variant: MatmulAllReduceAddRmsNormVariant = VARIANT,
    artifacts_dir: Path,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    world_size = int(os.environ.get("PTO_MC2_MM_ARN_WORLD_SIZE", variant.expected_world_size))
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
            "reason": launch.get("reason", "Distributed PTO matmul_all_reduce_add_rms_norm launch failed."),
            "variant": variant.as_dict(),
            "world_size": world_size,
            "rank_reports": launch.get("rank_reports", []),
        }

    rank_reports = launch["rank_reports"]
    per_rank_medians = [report["timings_ms"]["median"] for report in rank_reports]
    y_diff = max(report["correctness"]["y_max_abs_diff"] for report in rank_reports)
    norm_diff = max(report["correctness"]["norm_max_abs_diff"] for report in rank_reports)
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
            "y_max_abs_diff": y_diff,
            "norm_max_abs_diff": norm_diff,
            "max_abs_diff": max_abs_diff,
            "per_rank_max_abs_diff": [report["correctness"]["max_abs_diff"] for report in rank_reports],
        },
        "reference_contract": "pto_local_matmul_then_all_reduce_then_add_rms_norm",
        "rank_reports": rank_reports,
    }
