"""Runtime helpers for quant_grouped_matmul_inplace_add contract bring-up."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QuantGroupedMatmulInplaceAddVariant:
    groups: int
    m: int
    k: int
    n: int
    seed: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"g{self.groups}_m{self.m}_k{self.k}_n{self.n}"

    @property
    def shape_summary(self) -> dict[str, object]:
        return {
            "x1": [self.groups * 128, self.m],
            "x2": [self.groups * 128, self.n],
            "scale1": [self.groups],
            "scale2": [self.groups, self.n],
            "y": [self.groups, self.m, self.n],
            "group_list": [self.groups],
        }


VARIANT = QuantGroupedMatmulInplaceAddVariant(groups=4, m=96, k=128, n=128, seed=0)
VARIANTS = (VARIANT,)


def baseline_blocker(device_index: int = 0) -> dict[str, object]:
    del device_index
    return {
        "status": "blocked",
        "reason": (
            "This host does not expose a torch_npu Python entrypoint for quant_grouped_matmul_inplace_add. "
            "The upstream contract is currently only visible through ACLNN/C++ tests and examples."
        ),
        "entrypoint": "aclnnQuantGroupedMatmulInplaceAdd",
        "variants": [variant.as_dict() for variant in VARIANTS],
        "shape_summaries": [variant.shape_summary for variant in VARIANTS],
    }
