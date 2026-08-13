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

# ---- Orphan-binary guard -------------------------------------------------
#
# WHY THIS EXISTS. The loop below runs whatever executables it finds. A binary
# whose source and CMake target are gone still runs, still exits 0, and still
# prints plausible JSON -- which then becomes a committed reference that tests
# pin. One such binary once corrupted 13 reference keys, and another
# (cluster_w8b_creditriskplus_probe: no source dir, no CMake target, no git
# history at all) was still sitting in the build directory on 2026-08-11.
#
# A reference must be reproducible from committed source. So every binary we
# are about to run has to correspond to a target the current CMakeLists
# actually declares.

KNOWN_TARGETS=()
while IFS= read -r t; do
  KNOWN_TARGETS+=("$t")
done < <(cd "$PROBES_BUILD" && make help 2>/dev/null |
         sed -n 's/^\.\.\. \([A-Za-z0-9_]*_probe\)$/\1/p' | sort -u)

if [ "${#KNOWN_TARGETS[@]}" -eq 0 ]; then
  echo "ERROR: could not read any probe targets from $PROBES_BUILD."
  echo "The build directory looks unconfigured -- run ./build-cpp.sh first."
  exit 1
fi

is_known_target() {
  local needle="$1" t
  for t in "${KNOWN_TARGETS[@]}"; do
    [ "$t" = "$needle" ] && return 0
  done
  return 1
}

orphans=()
for probe in "$PROBES_BUILD"/*_probe; do
  [ -x "$probe" ] || continue
  is_known_target "$(basename "$probe")" || orphans+=("$(basename "$probe")")
done

if [ "${#orphans[@]}" -gt 0 ]; then
  echo "ERROR: ${#orphans[@]} orphan probe binary/binaries in $PROBES_BUILD --"
  echo "present on disk but declared by no current CMake target:"
  printf '  %s\n' "${orphans[@]}"
  echo ""
  echo "Anything they emit has no reproducible provenance. Delete them, or"
  echo "restore the source and its add_executable(), before regenerating."
  exit 1
fi

# ---- Run the probes ------------------------------------------------------

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
  # Write via a temporary so a probe that dies part-way cannot leave a
  # truncated reference behind that looks like a real one.
  tmp="$(mktemp)"
  if ! "$probe" > "$tmp"; then
    rm -f "$tmp"
    echo "ERROR: probe $name exited non-zero. $out_path left unchanged."
    exit 1
  fi
  mv "$tmp" "$out_path"
done

echo ""
echo "Done. JSONs under: $REFS_DIR"
