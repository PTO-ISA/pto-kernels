"""Runtime helpers for grouped_matmul_finalize_routing contract bring-up."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pto_kernels.utils.env import REQUIRED_TBE_PYTHON_MODULES, detect_env


@dataclass(frozen=True)
class GroupedMatmulFinalizeRoutingVariant:
    m: int
    k: int
    n: int
    weight_n_storage: int
    input_dtype: str
    weight_dtype: str
    seed: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"m{self.m}_k{self.k}_n{self.n}_{self.weight_dtype}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x": [self.m, self.k],
            "weight": [1, self.k, self.weight_n_storage],
            "scale": [1, 1, self.n],
            "pertoken_scale": [self.m],
            "logit": [self.m],
            "row_index": [self.m],
            "group_list": [1],
            "output": [self.m, self.n],
        }


VARIANT = GroupedMatmulFinalizeRoutingVariant(
    m=64,
    k=128,
    n=128,
    weight_n_storage=128,
    input_dtype="int8",
    weight_dtype="int8_nd",
    seed=0,
)

VARIANTS = (
    VARIANT,
    GroupedMatmulFinalizeRoutingVariant(
        m=64,
        k=192,
        n=1024,
        weight_n_storage=1024 // 8,
        input_dtype="int8",
        weight_dtype="int32_w4a8_nd",
        seed=1,
    ),
)


def baseline_blocker(device_index: int = 0) -> dict[str, object]:
    del device_index
    modules = detect_env().tbe_python_modules
    missing_modules = [name for name in REQUIRED_TBE_PYTHON_MODULES if not modules.get(name, False)]
    return {
        "status": "blocked",
        "reason": (
            "torch_npu.npu_grouped_matmul_finalize_routing on this host requires the routed quantized "
            "contract and either a valid ND quantized weight dtype or FRACTAL_NZ-style storage metadata. "
            "Measured probes show plain ND int8 weights are rejected, W4A8-sized ND probes can segfault, "
            "and the NZ-format path is still gated until local TBE-side format tooling is fully healthy."
        ),
        "entrypoint": "torch_npu.npu_grouped_matmul_finalize_routing",
        "environment": {
            "symbol_available": True,
            "quantized_route_required": True,
            "missing_tbe_python_modules": missing_modules,
            "measured_failures": [
                "missing scale/pertokenScale/logit/rowIndex -> invalid parameter",
                "plain ND int8 weight -> weightNd weight type should be INT_32 or supported low-precision format",
                "W4A8-sized ND probe -> host runtime segmentation fault before stable benchmark",
                "npu_format_cast NZ probe -> local TBE-side format materialization still not green",
            ],
        },
        "variants": [variant.as_dict() for variant in VARIANTS],
        "shape_summaries": [variant.shape_summary for variant in VARIANTS],
    }
