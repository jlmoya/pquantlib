#!/usr/bin/env bash
# Every probe must emit byte-identical JSON on repeated runs.
#
# WHY THIS EXISTS. A probe that reads uninitialised or out-of-bounds memory
# still exits 0 and still prints plausible-looking JSON. That JSON gets
# committed as a reference and pinned by tests, and the tests pass — against
# noise. Two such probes were found by accident, both after the references had
# already been committed:
#
#   * the barrier probe indexed vanilla()[8] on an 8-node grid; three runs gave
#     1.3e-114, 0 and 0 for the same key
#   * LossDistBinomial::volume_ is never assigned in C++ and is read as a loop
#     guard; 25 runs gave a different denormal near 2.14e-314 each time
#
# Nothing in the harness checked for this. A non-deterministic reference is
# worse than a missing one: it looks authoritative. Every cross-validated
# number in this port assumes probes are reproducible, so that assumption is
# now tested rather than trusted.
#
# Usage:
#   ./check_probe_determinism.sh <probe-build-dir> [runs]
#
# Exit 0 iff every probe that runs at all is byte-identical across runs.

set -uo pipefail

BUILD_DIR="${1:?usage: check_probe_determinism.sh <probe-build-dir> [runs]}"
RUNS="${2:-3}"
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DYLD_LIBRARY_PATH="$HARNESS_DIR/cpp/build/quantlib/ql:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$HARNESS_DIR/cpp/build/quantlib/ql:${LD_LIBRARY_PATH:-}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

nondeterministic=(); crashed=(); ok=0

while IFS= read -r probe; do
    name="$(basename "$probe")"
    first=""
    status="ok"
    for i in $(seq 1 "$RUNS"); do
        if ! "$probe" > "$TMP/$name.$i" 2>"$TMP/$name.$i.err"; then
            status="crash"; break
        fi
        if [ -z "$first" ]; then
            first="$TMP/$name.$i"
        elif ! cmp -s "$first" "$TMP/$name.$i"; then
            status="nondet"; break
        fi
    done
    case "$status" in
        ok)     ok=$((ok+1)) ;;
        crash)  crashed+=("$name") ;;
        nondet) nondeterministic+=("$name")
                echo "NON-DETERMINISTIC: $name"
                diff <(head -c 4000 "$first") <(head -c 4000 "$TMP/$name.$i") | head -6 ;;
    esac
done < <(find "$BUILD_DIR" -type f -perm +111 -name '*_probe' | sort)

echo
echo "deterministic:      $ok"
echo "non-deterministic:  ${#nondeterministic[@]}  ${nondeterministic[*]:-}"
echo "crashed / no-run:   ${#crashed[@]}  ${crashed[*]:-}"

[ "${#nondeterministic[@]}" -eq 0 ]
