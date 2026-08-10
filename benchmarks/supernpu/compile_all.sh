#!/bin/bash
# Top-level compilation script for the maintained PTO workload backend.

ARCH=${1:-one-level}

echo "=========================================="
echo "SuperNPUBench Build System"
echo "Architecture backend: $ARCH"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

compile_one_level() {
    echo ""
    echo ">>> Compiling one-level-arch backend..."
    if [ -f "$SCRIPT_DIR/benchmark/one-level-arch/compile_all.sh" ]; then
        bash "$SCRIPT_DIR/benchmark/one-level-arch/compile_all.sh"
    else
        echo "Warning: benchmark/one-level-arch/compile_all.sh not found"
    fi
}

case $ARCH in
    one-level|one-level-arch|pto|all)
        compile_one_level
        ;;
    *)
        echo "Usage: $0 [one-level|all]"
        echo "  one-level  - Compile one-level-arch backend only (benchmark/one-level-arch)"
        echo "  all        - Alias for the maintained backend"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Build completed for: $ARCH"
echo "=========================================="
