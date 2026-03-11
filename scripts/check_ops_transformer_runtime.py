#!/usr/bin/env python3
"""Inspect whether ops-transformer seed packages are built or installed."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from pto_kernels.utils import detect_env, inspect_ops_transformer_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print the runtime status as JSON.")
    args = parser.parse_args()

    env = detect_env()
    status = inspect_ops_transformer_runtime(toolkit_home=env.toolkit_home)
    if args.json:
        print(status.to_json())
        return 0

    print(f"ops_transformer_root : {status.ops_transformer_root}")
    print(f"toolkit_home         : {status.toolkit_home}")
    print(f"install_root         : {status.install_root}")
    print(f"build_out            : {status.build_out}")
    print(f"package_path         : {status.package_path}")
    print(f"build_dep_metadata   : {status.build_dependency_metadata_present}")
    print(f"package_installed    : {status.package_installed}")
    print(f"vendor_packages      : {status.vendor_packages_present}")
    print(f"seed_ops             : {','.join(status.seed_ops)}")
    if status.package_runfiles:
        print("package_runfiles:")
        for runfile in status.package_runfiles:
            print(f"  - {runfile}")
    if status.uninstall_script:
        print(f"uninstall_script     : {status.uninstall_script}")
    if status.vendors_config:
        print(f"vendors_config       : {status.vendors_config}")
    if status.binary_info_configs:
        print("binary_info_configs:")
        for path in status.binary_info_configs:
            print(f"  - {path}")
    if status.missing_version_infos:
        print("missing_version_infos:")
        for path in status.missing_version_infos:
            print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
