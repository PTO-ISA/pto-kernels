"""Runtime helpers for grouped_matmul_swiglu_quant contract bring-up."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

from pto_kernels.utils.env import REQUIRED_TBE_PYTHON_MODULES, detect_env


@dataclass(frozen=True)
class GroupedMatmulSwigluQuantVariant:
    m: int
    k: int
    n: int
    e: int
    input_dtype: str
    weight_dtype: str
    seed: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"m{self.m}_k{self.k}_n{self.n}_e{self.e}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x": [self.m, self.k],
            "weight_logical": [self.e, self.k, self.n],
            "weight_storage_nz": [self.e, self.n // 32, self.k // 16, 16, 32],
            "weight_scale": [self.e, self.n],
            "x_scale": [self.m],
            "group_list": [self.e],
            "output": [self.m, self.n // 2],
            "output_scale": [self.m],
        }


VARIANT = GroupedMatmulSwigluQuantVariant(
    m=1024,
    k=256,
    n=4096,
    e=16,
    input_dtype="int8",
    weight_dtype="int8_fractal_nz",
    seed=0,
)
VARIANTS = (VARIANT,)


def _blocked(reason: str, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "blocked",
        "reason": reason,
        "entrypoint": "torch_npu.npu_grouped_matmul_swiglu_quant",
        "variants": [variant.as_dict() for variant in VARIANTS],
        "shape_summaries": [variant.shape_summary for variant in VARIANTS],
    }
    payload.update(extra)
    return payload


def _ok(**extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "ok",
        "entrypoint": "torch_npu.npu_grouped_matmul_swiglu_quant",
        "variants": [variant.as_dict() for variant in VARIANTS],
        "shape_summaries": [variant.shape_summary for variant in VARIANTS],
    }
    payload.update(extra)
    return payload


def _missing_tbe_modules() -> list[str]:
    modules = detect_env().tbe_python_modules
    return [name for name in REQUIRED_TBE_PYTHON_MODULES if not modules.get(name, False)]


def _group_list(m: int, e: int, device: str):
    import torch

    counts = [m // e] * e
    counts[-1] += m - sum(counts)
    return torch.tensor(counts, dtype=torch.int64, device=device).cumsum(dim=0)


def probe_baseline_contract(device_index: int = 0) -> dict[str, object]:
    missing = _missing_tbe_modules()
    if missing:
        return _blocked(
            "Host baseline still depends on local TBE Python modules before NZ-format materialization can be probed.",
            missing_tbe_python_modules=missing,
        )

    try:
        import torch
        import torch_npu
    except Exception as exc:  # pragma: no cover - environment error
        return _blocked(f"Unable to import torch_npu stack: {type(exc).__name__}: {exc}")

    if not hasattr(torch_npu, "npu_grouped_matmul_swiglu_quant"):
        return _blocked("torch_npu does not expose npu_grouped_matmul_swiglu_quant")

    variant = VARIANT
    device = f"npu:{device_index}"
    physical_weight_shape = (variant.e, variant.n // 32, variant.k // 16, 16, 32)
    try:
        x = torch.randint(-8, 8, (variant.m, variant.k), dtype=torch.int8, device=device)
        weight_storage = torch.randint(-8, 8, physical_weight_shape, dtype=torch.int8, device=device)
        weight_nz = torch_npu.npu_format_cast(weight_storage, 29)
        weight_format = int(torch_npu.get_npu_format(weight_nz))
        weight_scale = torch.full((variant.e, variant.n), 1.0, dtype=torch.float32, device=device)
        x_scale = torch.full((variant.m,), 1.0, dtype=torch.float32, device=device)
        group_list = _group_list(variant.m, variant.e, device=device)
    except Exception as exc:
        return _blocked(
            f"Unable to materialize NZ-format weight storage on this host: {type(exc).__name__}: {exc}",
            attempted_weight_storage_shape=list(physical_weight_shape),
        )

    if weight_format != 29:
        return _blocked(
            "npu_format_cast completed but did not materialize FRACTAL_NZ format 29.",
            observed_weight_format=weight_format,
            attempted_weight_storage_shape=list(physical_weight_shape),
        )

    try:
        start = time.perf_counter()
        output, output_scale, output_offset = torch_npu.npu_grouped_matmul_swiglu_quant(
            x,
            weight_nz,
            group_list,
            weight_scale,
            x_scale,
        )
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1e3
        return _ok(
            probe_variant=variant.as_dict(),
            measured_weight_format=weight_format,
            output_shape=list(output.shape),
            output_scale_shape=list(output_scale.shape),
            output_offset_shape=list(output_offset.shape),
            probe_latency_ms=elapsed_ms,
        )
    except Exception as exc:
        return _blocked(
            f"NZ-format materialization succeeded but baseline op call still failed: {type(exc).__name__}: {exc}",
            probe_variant=variant.as_dict(),
            measured_weight_format=weight_format,
            attempted_weight_storage_shape=list(physical_weight_shape),
        )


def baseline_blocker(device_index: int = 0) -> dict[str, object]:
    report = probe_baseline_contract(device_index=device_index)
    if report["status"] == "ok":
        return report
    return report
