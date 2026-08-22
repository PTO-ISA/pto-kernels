#!/bin/bash
# Build one or every active microbenchmark category with aggregate failures.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICROBENCH_ROOT=${MICROBENCH_ROOT:-$SCRIPT_DIR}
CATEGORY=${1:-all}

run_category() {
    local category=$1
    local label=$2
    local root="$MICROBENCH_ROOT/$category"

    echo ""
    echo ">>> $label"
    if [ ! -d "$root" ]; then
        echo "ERROR: microbenchmark category directory is missing: $root"
        return 1
    fi
    if [ ! -f "$root/compile.all" ]; then
        echo "ERROR: microbenchmark category has no compile.all: $root"
        return 1
    fi
    if (cd "$root" && bash compile.all); then
        echo "PASS: $category category"
        return 0
    fi
    echo "FAIL: $category category"
    return 1
}

selected=()
case "$CATEGORY" in
    cube|vector|memory|scalar)
        selected+=("$CATEGORY")
        ;;
    all)
        selected+=(cube vector memory scalar)
        ;;
    *)
        echo "Usage: $0 [cube|vector|memory|scalar|all]"
        exit 1
        ;;
esac

failures=()
for category in "${selected[@]}"; do
    case "$category" in
        cube) label="matrix (CUBE)" ;;
        vector) label="VEC/SFU" ;;
        memory) label="memory (TLSU)" ;;
        scalar) label="scalar (GPR ALU)" ;;
    esac
    if ! run_category "$category" "$label"; then
        failures+=("$category")
    fi
done

if [ "${#failures[@]}" -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "Microbench build FAILED: ${failures[*]}"
    echo "=========================================="
    exit 1
fi

echo ""
echo "=========================================="
echo "Microbench build completed: all selected categories passed ($CATEGORY)"
echo "=========================================="
