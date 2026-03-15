#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${ASCEND_TOOLKIT_HOME:-}" && -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]]; then
  TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME}"
elif [[ -f "${HOME}/Ascend/cann/set_env.sh" ]]; then
  TOOLKIT_HOME="${HOME}/Ascend/cann"
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

CUSTOM_OP_ENV="${TOOLKIT_HOME}/vendors/custom_transformer/bin/set_env.bash"
if [[ -f "${CUSTOM_OP_ENV}" ]]; then
  export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
  # shellcheck source=/dev/null
  source "${CUSTOM_OP_ENV}"
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

if [[ -n "${PTO_PTO_ISA_ROOT:-}" ]]; then
  export PTO_ISA_ROOT="${PTO_PTO_ISA_ROOT}"
  export PTO_ISA_INCLUDE_DIR="${PTO_PTO_ISA_ROOT}/include"
  export CPATH="${PTO_ISA_INCLUDE_DIR}:${CPATH:-}"
fi

if [[ -n "${PTO_PTO_DSL_ROOT:-}" ]]; then
  export PYTHONPATH="${PTO_PTO_DSL_ROOT}:${REPO_ROOT}/python:${PYTHONPATH:-}"
else
  export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
fi

ptoas_candidates=()
if [[ -n "${PTO_PTOAS_ROOT:-}" ]]; then
  ptoas_candidates+=(
    "${PTO_PTOAS_ROOT}/build-19/tools/ptoas"
    "${PTO_PTOAS_ROOT}/build/tools/ptoas"
  )
fi
ptoas_candidates+=(
  "${REPO_ROOT}/../PTOAS/build-19/tools/ptoas"
  "${REPO_ROOT}/../PTOAS/build/tools/ptoas"
  "${REPO_ROOT}/external/src/PTOAS/build-19/tools/ptoas"
  "${REPO_ROOT}/external/src/PTOAS/build/tools/ptoas"
)

for ptoas_dir in "${ptoas_candidates[@]}"; do
  if [[ -x "${ptoas_dir}/ptoas" ]]; then
    export PATH="${ptoas_dir}:${PATH}"
    break
  fi
done

echo "[env] PTO_KERNELS_SOC=${PTO_KERNELS_SOC}"
echo "[env] PTO_KERNELS_PTO_ARCH=${PTO_KERNELS_PTO_ARCH}"
echo "[env] PTO_KERNELS_NPU_ARCH=${PTO_KERNELS_NPU_ARCH}"
echo "[env] PTO_ISA_INCLUDE_DIR=${PTO_ISA_INCLUDE_DIR:-}"
echo "[env] PYTHONPATH=${PYTHONPATH}"
