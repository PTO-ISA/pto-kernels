"""Helpers for ops-transformer seed-package bring-up on the local workspace."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from pto_kernels.config import resolve_workspace_repo


SEED_OPS = [
    "apply_rotary_pos_emb",
    "grouped_matmul",
    "ffn",
    "moe_token_permute",
    "flash_attention_score",
    "matmul_reduce_scatter",
]

REQUIRED_BUILD_INFO_PACKAGES = [
    "runtime",
    "opbase",
    "hcomm",
    "ge-executor",
    "metadef",
    "ge-compiler",
    "asc-devkit",
    "bisheng-compiler",
    "asc-tools",
]


def _infer_install_root(toolkit_home: str | None) -> Path | None:
    if not toolkit_home:
        return None
    path = Path(toolkit_home).resolve()
    parts = path.parts
    if "ascend-toolkit" in parts:
        idx = parts.index("ascend-toolkit")
        if idx > 0:
            return Path(*parts[:idx])
    return path.parent


def _discover_runfiles(build_out: Path) -> list[Path]:
    if not build_out.exists():
        return []
    return sorted(build_out.glob("cann-*-ops-transformer_*_linux-*.run"))


def required_version_info_paths(package_path: str | None) -> list[Path]:
    if not package_path:
        return []
    root = Path(package_path)
    return [root / "share" / "info" / pkg / "version.info" for pkg in REQUIRED_BUILD_INFO_PACKAGES]


@dataclass
class OpsTransformerRuntimeStatus:
    ops_transformer_root: str | None
    toolkit_home: str | None
    install_root: str | None
    build_out: str | None
    seed_ops: list[str]
    package_runfiles: list[str]
    package_path: str | None
    share_info_dir: str | None
    uninstall_script: str | None
    vendors_dir: str | None
    vendors_config: str | None
    binary_info_configs: list[str]
    required_version_infos: list[str]
    missing_version_infos: list[str]
    build_dependency_metadata_present: bool
    package_installed: bool
    vendor_packages_present: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def inspect_ops_transformer_runtime(*, toolkit_home: str | None) -> OpsTransformerRuntimeStatus:
    ops_root = resolve_workspace_repo("ops-transformer")
    install_root = _infer_install_root(toolkit_home)

    build_out = ops_root / "build_out" if ops_root else None
    package_runfiles = _discover_runfiles(build_out) if build_out else []

    share_info_dir = install_root / "share" / "info" / "ops_transformer" if install_root else None
    uninstall_script = share_info_dir / "script" / "uninstall.sh" if share_info_dir else None
    vendors_dir = Path(toolkit_home) / "opp" / "vendors" if toolkit_home else None
    vendors_config = vendors_dir / "config.ini" if vendors_dir else None
    binary_info_configs = []
    if vendors_dir and vendors_dir.exists():
        binary_info_configs = sorted(str(path) for path in vendors_dir.rglob("binary_info_config.json"))
    required_infos = required_version_info_paths(toolkit_home)
    missing_infos = [str(path) for path in required_infos if not path.exists()]

    return OpsTransformerRuntimeStatus(
        ops_transformer_root=str(ops_root) if ops_root else None,
        toolkit_home=toolkit_home,
        install_root=str(install_root) if install_root else None,
        build_out=str(build_out) if build_out and build_out.exists() else None,
        seed_ops=list(SEED_OPS),
        package_runfiles=[str(path) for path in package_runfiles],
        package_path=toolkit_home,
        share_info_dir=str(share_info_dir) if share_info_dir and share_info_dir.exists() else None,
        uninstall_script=str(uninstall_script) if uninstall_script and uninstall_script.exists() else None,
        vendors_dir=str(vendors_dir) if vendors_dir and vendors_dir.exists() else None,
        vendors_config=str(vendors_config) if vendors_config and vendors_config.exists() else None,
        binary_info_configs=binary_info_configs,
        required_version_infos=[str(path) for path in required_infos],
        missing_version_infos=missing_infos,
        build_dependency_metadata_present=not missing_infos,
        package_installed=bool(uninstall_script and uninstall_script.exists()),
        vendor_packages_present=bool(binary_info_configs),
    )


__all__ = [
    "OpsTransformerRuntimeStatus",
    "REQUIRED_BUILD_INFO_PACKAGES",
    "SEED_OPS",
    "inspect_ops_transformer_runtime",
    "required_version_info_paths",
]
