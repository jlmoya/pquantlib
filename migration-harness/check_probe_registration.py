#!/usr/bin/env python3
"""Every declared probe must have a source, and every source must be declared.

WHY THIS EXISTS
---------------
The probe CMakeLists registers most probes explicitly but registers several
families through ``foreach(area ...)`` loops, and three of those loops wrap the
``add_executable`` in ``if(EXISTS .../probe.cpp)``. That guard means a declared
area whose source was never written produces **no CMake target at all** -- so it
is invisible to a check that compares targets against binaries, and invisible to
the orphan-binary guard in generate-references.sh. It simply disappears.

Measured on 2026-08-29: two areas (v143_experimental_crediteng and
v143_ts_spreadcurve) had been declared in guarded loops and never written. No
directory, no reference, no consumer -- the CMakeLists claimed coverage the
harness did not have.

The reverse gap matters too: a probe source sitting on disk that no target
declares never runs, so whatever it was written to cross-validate is not
actually pinned by anything.

Exit 0 iff both directions are clean.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROBES = Path(__file__).resolve().parent / "cpp" / "probes"
CMAKELISTS = PROBES / "CMakeLists.txt"

# foreach(area a b c)\n  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/<prefix>${area}/probe.cpp)
GUARDED = re.compile(
    r"foreach\(area([^)]*)\)\s*\n\s*"
    r"if\(EXISTS \$\{CMAKE_CURRENT_SOURCE_DIR\}/([A-Za-z0-9_]*)\$\{area\}/probe\.cpp\)"
)
# foreach(area a b c)\n  add_executable(<prefix>${area}_probe <prefix>${area}/probe.cpp)
UNGUARDED = re.compile(
    r"foreach\(area([^)]*)\)\s*\n\s*"
    r"add_executable\([A-Za-z0-9_]*\$\{area\}_probe\s+([A-Za-z0-9_]*)\$\{area\}/probe\.cpp\)"
)
EXPLICIT = re.compile(r"^\s*add_executable\(\s*([A-Za-z0-9_]+)\s+([^\s)]+)\s*\)", re.M)


def declared_sources(src: str) -> dict[str, str]:
    """Map declared source path -> how it was declared."""
    out: dict[str, str] = {}
    for m in EXPLICIT.finditer(src):
        if "${" in m.group(2):
            continue  # loop bodies are handled below
        out[m.group(2)] = "explicit"
    for areas, prefix in GUARDED.findall(src):
        for a in areas.split():
            out[f"{prefix}{a}/probe.cpp"] = "foreach + if(EXISTS)"
    for areas, prefix in UNGUARDED.findall(src):
        for a in areas.split():
            out[f"{prefix}{a}/probe.cpp"] = "foreach"
    return out


def main() -> int:
    if not CMAKELISTS.is_file():
        print(f"ERROR: {CMAKELISTS} not found", file=sys.stderr)
        return 2

    src = CMAKELISTS.read_text()
    declared = declared_sources(src)

    missing = sorted(p for p in declared if not (PROBES / p).is_file())
    on_disk = {
        str(p.relative_to(PROBES))
        for p in PROBES.rglob("*.cpp")
        if "build" not in p.parts
    }
    unregistered = sorted(on_disk - set(declared))

    print(f"declared probe sources : {len(declared)}")
    print(f"probe sources on disk  : {len(on_disk)}")

    if missing:
        print(f"\nDECLARED BUT MISSING ({len(missing)}) -- registered in CMake, no source:")
        for p in missing:
            note = declared[p]
            hint = " [silently skipped by the guard]" if "EXISTS" in note else " [hard CMake error]"
            print(f"   {p}   ({note}){hint}")

    if unregistered:
        print(f"\nON DISK BUT UNREGISTERED ({len(unregistered)}) -- never built, so it pins nothing:")
        for p in unregistered:
            print(f"   {p}")

    if missing or unregistered:
        print("\nFAIL: probe registration and probe sources disagree.")
        return 1

    print("\nOK: every declared probe has a source, and every source is declared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
