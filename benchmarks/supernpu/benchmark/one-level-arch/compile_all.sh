#!/bin/bash
# Compile every active one-level operator and fail if any row fails.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${COMPILER_DIR:?Set COMPILER_DIR to the in-repo Linx compiler bin directory}"
export COMPILER_DIR
REPO_ROOT=${REPO_ROOT:-$SCRIPT_DIR}

echo "=========================================="
echo "[PTO ISA] Starting active full compilation"
echo "REPO_ROOT: $REPO_ROOT"
echo "=========================================="

compile_operator() {
    local operator_path=$1
    local operator_name=$2

    echo ""
    echo "------------------------------------------"
    echo "Compiling: $operator_name"
    echo "Path: $operator_path"
    echo "------------------------------------------"

    if [ ! -d "$operator_path" ]; then
        echo "ERROR: active operator directory not found: $operator_path"
        return 1
    fi
    if [ ! -f "$operator_path/compile.all" ]; then
        echo "ERROR: active operator has no compile.all: $operator_path"
        return 1
    fi

    if (
        cd "$operator_path" || exit 1
        export baremetal=${baremetal:-off}
        bash compile.all
    ); then
        echo "PASS: $operator_name compilation"
        return 0
    fi
    echo "FAIL: $operator_name compilation"
    return 1
}

failures=()
while IFS='|' read -r operator_path operator_name; do
    if ! compile_operator "$REPO_ROOT/test/kernel/$operator_path" "$operator_name"; then
        failures+=("$operator_name")
    fi
done <<'EOF'
matmul|matmul
broadcast|broadcast
concat|concat
gather|gather
transpose|transpose
element_wise/gelu|gelu
reduction/reducemax_col|reducemax_col
reduction/reducemax_row|reducemax_row
reduction/reducesum_col|reducesum_col
reduction/reducesum_row|reducesum_row
control|control
sort|sort
deepseek|deepseek
EOF

if [ "${#failures[@]}" -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "Full compilation FAILED: ${failures[*]}"
    echo "=========================================="
    exit 1
fi

echo ""
echo "=========================================="
echo "Full compilation completed: all active operators passed"
echo "=========================================="
echo ""
echo "Generated ELF files:"
if [ -d "$REPO_ROOT/output" ]; then
    find "$REPO_ROOT/output" -name "*.elf" -type f | wc -l
else
    echo 0
fi
echo "ELF files are located in: $REPO_ROOT/output/"
