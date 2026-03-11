#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/source_env.sh" >/dev/null

OPS_ROOT="${PTO_OPS_TRANSFORMER_ROOT:-${REPO_ROOT}/../ops-transformer}"
if [[ ! -d "${OPS_ROOT}" ]]; then
  echo "[ERROR] Unable to locate ops-transformer workspace at ${OPS_ROOT}" >&2
  exit 1
fi

SEED_OPS="apply_rotary_pos_emb,grouped_matmul,ffn,moe_token_permute,flash_attention_score,matmul_reduce_scatter"
OPS="${SEED_OPS}"
SOC="${PTO_KERNELS_SOC}"
PACKAGE_PATH="${ASCEND_TOOLKIT_HOME:-}"
INSTALL_ROOT="$(python3 - <<'PY'
from pathlib import Path
import os
toolkit_home = os.environ.get("ASCEND_TOOLKIT_HOME", "")
path = Path(toolkit_home).resolve()
parts = path.parts
if "ascend-toolkit" in parts:
    idx = parts.index("ascend-toolkit")
    print(Path(*parts[:idx]))
else:
    print(path.parent)
PY
)"
DO_INSTALL=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ops)
      OPS="$2"
      shift 2
      ;;
    --ops=*)
      OPS="${1#*=}"
      shift
      ;;
    --soc)
      SOC="$2"
      shift 2
      ;;
    --soc=*)
      SOC="${1#*=}"
      shift
      ;;
    --install)
      DO_INSTALL=1
      shift
      ;;
    --package-path)
      PACKAGE_PATH="$2"
      shift 2
      ;;
    --package-path=*)
      PACKAGE_PATH="${1#*=}"
      shift
      ;;
    --install-root)
      INSTALL_ROOT="$2"
      shift 2
      ;;
    --install-root=*)
      INSTALL_ROOT="${1#*=}"
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

echo "[bringup] ops_root=${OPS_ROOT}"
echo "[bringup] soc=${SOC}"
echo "[bringup] ops=${OPS}"
echo "[bringup] package_path=${PACKAGE_PATH}"
echo "[bringup] install_root=${INSTALL_ROOT}"

if [[ -z "${PACKAGE_PATH}" ]]; then
  echo "[ERROR] ASCEND toolkit package path is empty. Source the env first or pass --package-path." >&2
  exit 1
fi

export PACKAGE_PATH
python3 - <<'PY'
import os
import sys
from pathlib import Path

package_path = os.environ["PACKAGE_PATH"]
required = [
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
missing = [str(Path(package_path) / "share" / "info" / pkg / "version.info") for pkg in required
           if not (Path(package_path) / "share" / "info" / pkg / "version.info").exists()]
if missing:
    print("[ERROR] ops-transformer package build prerequisites are missing under --package-path:", file=sys.stderr)
    for path in missing:
        print(f"  - {path}", file=sys.stderr)
    sys.exit(2)
PY

cd "${OPS_ROOT}"
bash build.sh --pkg "--soc=${SOC}" "--ops=${OPS}" "--package-path=${PACKAGE_PATH}" "${EXTRA_ARGS[@]}"

RUNFILE="$(find "${OPS_ROOT}/build_out" -maxdepth 1 -type f -name 'cann-*-ops-transformer_*_linux-*.run' | sort | tail -n 1)"
if [[ -z "${RUNFILE}" ]]; then
  echo "[ERROR] build completed but no ops-transformer runfile was found under build_out" >&2
  exit 1
fi

echo "[bringup] runfile=${RUNFILE}"
if [[ "${DO_INSTALL}" -eq 1 ]]; then
  "${RUNFILE}" --full "--install-path=${INSTALL_ROOT}"
  echo "[bringup] installation finished"
fi
