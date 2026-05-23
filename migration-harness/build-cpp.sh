#!/usr/bin/env bash
# Build QuantLib v1.42.1 submodule + all probe binaries.
# Run once after `git submodule update --init`.

set -euo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QL_SRC="$HARNESS_DIR/cpp/quantlib"
QL_BUILD="$HARNESS_DIR/cpp/build/quantlib"
PROBES_BUILD="$HARNESS_DIR/cpp/build/probes"

if [ ! -d "$QL_SRC" ]; then
  echo "ERROR: QuantLib submodule not initialized."
  echo "Run: git submodule update --init --recursive"
  exit 1
fi

mkdir -p "$QL_BUILD" "$PROBES_BUILD"

echo "=== Building QuantLib v1.42.1 ==="
cd "$QL_BUILD"
cmake "$QL_SRC" \
  -DCMAKE_BUILD_TYPE=Release \
  -DQL_BUILD_EXAMPLES=OFF \
  -DQL_BUILD_BENCHMARK=OFF \
  -DQL_BUILD_TEST_SUITE=OFF
cmake --build . --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

echo ""
echo "=== Building probes ==="
cd "$PROBES_BUILD"
cmake "$HARNESS_DIR/cpp/probes" \
  -DCMAKE_BUILD_TYPE=Release \
  -DQUANTLIB_BUILD_DIR="$QL_BUILD" \
  -DQUANTLIB_SRC_DIR="$QL_SRC"
cmake --build . --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

echo ""
echo "=== Build complete ==="
echo "QuantLib library: $QL_BUILD/ql/libQuantLib.dylib (or .so on Linux)"
echo "Probe binaries:    $PROBES_BUILD/<topic>_<class>_probe"
echo ""
echo "Next: ./migration-harness/generate-references.sh"
