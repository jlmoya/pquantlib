#!/usr/bin/env bash
# Run all probes, regenerate reference JSONs.
#
# Usage:
#   ./generate-references.sh           # run all probes
#   ./generate-references.sh <pattern> # run only probes matching pattern

set -euo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBES_BUILD="$HARNESS_DIR/cpp/build/probes"
REFS_DIR="$HARNESS_DIR/references"

if [ ! -d "$PROBES_BUILD" ]; then
  echo "ERROR: probes not built. Run ./build-cpp.sh first."
  exit 1
fi

PATTERN="${1:-*}"

mkdir -p "$REFS_DIR"

for probe in "$PROBES_BUILD"/*_probe; do
  [ -x "$probe" ] || continue
  name="$(basename "$probe" _probe)"
  if [[ "$name" != $PATTERN ]]; then
    continue
  fi
  # Derive output path: probe name "math_copulas_clayton" → "references/math/copulas/clayton.json"
  out_subpath="$(echo "$name" | tr '_' '/')"
  out_path="$REFS_DIR/$out_subpath.json"
  mkdir -p "$(dirname "$out_path")"
  echo "running $name → $out_path"
  "$probe" > "$out_path"
done

echo ""
echo "Done. JSONs under: $REFS_DIR"
