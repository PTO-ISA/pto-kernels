"""Runtime helpers for grouped_matmul_swiglu_quant_v2 contract bring-up."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GroupedMatmulSwigluQuantV2Variant:
    m: int
    k: int
    n: int
    e: int
    seed: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"m{self.m}_k{self.k}_n{self.n}_e{self.e}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x": [self.m, self.k],
            "weight_list_item": [self.e, self.k, self.n],
            "weight_scale_list_item": [self.e, self.k // 64, self.n, 2],
            "x_scale": [self.m, self.k // 64, 2],
            "group_list": [self.e],
            "output": [self.m, self.n // 2],
            "output_scale": [self.m, (self.n // 2) // 64, 2],
        }


VARIANT = GroupedMatmulSwigluQuantV2Variant(m=2048, k=7168, n=4096, e=8, seed=0)
VARIANTS = (VARIANT,)


def baseline_blocker(device_index: int = 0) -> dict[str, object]:
    del device_index
    return {
        "status": "blocked",
        "reason": (
            "torch_npu.npu_grouped_matmul_swiglu_quant_v2 requires list-valued FP8/scale contracts matching the "
            "upstream ACLNN example. A stable minimal Python baseline slice is not reproduced yet on this host."
        ),
        "entrypoint": "torch_npu.npu_grouped_matmul_swiglu_quant_v2",
        "variants": [variant.as_dict() for variant in VARIANTS],
        "shape_summaries": [variant.shape_summary for variant in VARIANTS],
    }
