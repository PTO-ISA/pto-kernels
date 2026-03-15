#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

declare -A PACKAGE_MAP=(
  [decorator]="decorator"
  [attrs]="attrs"
  [scipy]="scipy"
  [psutil]="psutil"
  [cloudpickle]="cloudpickle"
  [tornado]="tornado"
  [absl]="absl-py"
)

mapfile -t MISSING_MODULES < <(
  python3 "${REPO_ROOT}/scripts/check_tbe_python_deps.py" --json \
    | python3 -c 'import json, sys; print("\n".join(json.load(sys.stdin)["missing_modules"]))'
)

if [[ "${#MISSING_MODULES[@]}" -eq 0 ]]; then
  echo "[tbe-python-deps] all required modules are already installed"
  exit 0
fi

PACKAGES=()
for module_name in "${MISSING_MODULES[@]}"; do
  package_name="${PACKAGE_MAP[$module_name]:-}"
  if [[ -n "${package_name}" ]]; then
    PACKAGES+=("${package_name}")
  fi
done

if [[ "${#PACKAGES[@]}" -eq 0 ]]; then
  echo "[tbe-python-deps] no installable package mapping found for missing modules: ${MISSING_MODULES[*]}" >&2
  exit 1
fi

echo "[tbe-python-deps] installing: ${PACKAGES[*]}"
python3 -m pip install --user "${PACKAGES[@]}"
