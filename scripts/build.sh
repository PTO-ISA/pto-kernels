#!/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

SHORT=v:,
LONG=soc-version:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
SOC_VERSION="Ascend910B4"

while :; do
    case "$1" in
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR] Unexpected option: $1"
        break
        ;;
    esac
done

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH="$ASCEND_INSTALL_PATH"
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH="$ASCEND_HOME_PATH"
else
    if [ -d "$HOME/Ascend/cann" ]; then
        _ASCEND_INSTALL_PATH="$HOME"/Ascend/cann
    elif [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH="$HOME"/Ascend/ascend-toolkit/latest
    elif [ -d /usr/local/Ascend/cann ]; then
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/cann
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
if [ -f "$_ASCEND_INSTALL_PATH"/set_env.sh ]; then
    # shellcheck source=/dev/null
    source "$_ASCEND_INSTALL_PATH"/set_env.sh
else
    # shellcheck source=/dev/null
    source "$_ASCEND_INSTALL_PATH"/bin/setenv.bash
fi
echo "Current compile soc version is ${SOC_VERSION}"

# See https://docs.pytorch.org/cppdocs/installing.html
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
export CMAKE_PREFIX_PATH


echo "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"


set -e
rm -rf build
mkdir -p build
cmake -S "${PARENT_DIR}" \
      -B build \
      -DSOC_VERSION="${SOC_VERSION}" \
      -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
      -DASCEND_CANN_PACKAGE_PATH="${_ASCEND_INSTALL_PATH}"

cmake --build build  -j
