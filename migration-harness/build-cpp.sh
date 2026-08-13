#!/usr/bin/env bash
# Build QuantLib v1.43 submodule + all probe binaries.
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

echo "=== Building QuantLib v1.43 ==="
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

# Keep going past a failure rather than dying on the first one.
#
# WHY THIS EXISTS. This script runs under `set -e`, so a single probe that
# failed to compile used to abort the entire build. Every probe later in the
# build order was then silently absent -- for reasons having nothing to do
# with its own health -- and generate-references.sh would happily run the
# partial set and emit a reference tree that looked complete.
#
# Measured on 2026-08-11: 107 binaries existed, 2 probes were actually
# broken, and 118 perfectly good probes had simply never been reached.
# Rebuilding with -k took the count to 225.
#
# Failures are collected and reported together at the end instead.
set +e
cmake --build . --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" -- -k
build_status=$?
set -e

# Every registered target must have produced a binary. This catches both a
# genuine compile failure and a target whose link step never ran.
missing=()
while IFS= read -r target; do
  [ -x "$PROBES_BUILD/$target" ] || missing+=("$target")
done < <(make help 2>/dev/null | sed -n 's/^\.\.\. \([A-Za-z0-9_]*_probe\)$/\1/p' | sort -u)

if [ "${#missing[@]}" -gt 0 ]; then
  echo ""
  echo "ERROR: ${#missing[@]} probe target(s) registered in CMake produced no binary:"
  printf '  %s\n' "${missing[@]}"
  echo ""
  echo "References generated now would silently cover only the probes that built."
  echo "Fix these, or remove their add_executable() with a stated reason."
  exit 1
fi

if [ "$build_status" -ne 0 ]; then
  echo ""
  echo "ERROR: the probe build reported failures (exit $build_status) even though"
  echo "every registered target has a binary. Investigate before generating"
  echo "references -- a stale binary can mask a source that no longer compiles."
  exit 1
fi

echo ""
echo "=== Build complete ==="
echo "QuantLib library: $QL_BUILD/ql/libQuantLib.dylib (or .so on Linux)"
echo "Probe binaries:    $PROBES_BUILD/<topic>_<class>_probe"
echo ""
echo "Next: ./migration-harness/generate-references.sh"
