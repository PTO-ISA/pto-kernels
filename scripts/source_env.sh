#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${ASCEND_TOOLKIT_HOME:-}" && -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]]; then
  TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME}"
elif [[ -f "${HOME}/Ascend/ascend-toolkit/set_env.sh" ]]; then
  TOOLKIT_HOME="${HOME}/Ascend/ascend-toolkit"
elif [[ -f "${HOME}/Ascend/ascend-toolkit/latest/set_env.sh" ]]; then
  TOOLKIT_HOME="${HOME}/Ascend/ascend-toolkit/latest"
elif [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]]; then
  TOOLKIT_HOME="/usr/local/Ascend/ascend-toolkit"
else
  echo "[ERROR] Unable to locate Ascend toolkit set_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ -f "${TOOLKIT_HOME}/set_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${TOOLKIT_HOME}/set_env.sh"
elif [[ -f "${TOOLKIT_HOME}/bin/setenv.bash" ]]; then
  # shellcheck source=/dev/null
  source "${TOOLKIT_HOME}/bin/setenv.bash"
fi

export PTO_KERNELS_ROOT="${REPO_ROOT}"
export PTO_KERNELS_SOC="ascend910b"
export PTO_KERNELS_PTO_ARCH="a3"
export PTO_KERNELS_NPU_ARCH="dav-2201"

for repo_name in pto-dsl PTOAS pto-isa ops-transformer; do
  upper_name="$(echo "${repo_name}" | tr '[:lower:]-' '[:upper:]_')"
  if [[ -d "${REPO_ROOT}/external/src/${repo_name}" ]]; then
    export "PTO_${upper_name}_ROOT=${REPO_ROOT}/external/src/${repo_name}"
  elif [[ -d "${REPO_ROOT}/../${repo_name}" ]]; then
    export "PTO_${upper_name}_ROOT=${REPO_ROOT}/../${repo_name}"
  fi
done

if [[ -n "${PTO_PTO_DSL_ROOT:-}" ]]; then
  export PYTHONPATH="${PTO_PTO_DSL_ROOT}:${REPO_ROOT}/python:${PYTHONPATH:-}"
else
  export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
fi

if command -v ptoas >/dev/null 2>&1; then
  :
elif [[ -x "${REPO_ROOT}/../PTOAS/build/tools/ptoas/ptoas" ]]; then
  export PATH="${REPO_ROOT}/../PTOAS/build/tools/ptoas:${PATH}"
elif [[ -x "${REPO_ROOT}/external/src/PTOAS/build/tools/ptoas/ptoas" ]]; then
  export PATH="${REPO_ROOT}/external/src/PTOAS/build/tools/ptoas:${PATH}"
fi

echo "[env] PTO_KERNELS_SOC=${PTO_KERNELS_SOC}"
echo "[env] PTO_KERNELS_PTO_ARCH=${PTO_KERNELS_PTO_ARCH}"
echo "[env] PTO_KERNELS_NPU_ARCH=${PTO_KERNELS_NPU_ARCH}"
echo "[env] PYTHONPATH=${PYTHONPATH}"
