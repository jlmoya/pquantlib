#!/usr/bin/env python3
"""Ground-truth coverage audit: every C++ QuantLib class vs every ported Python class.

Denominator = the pinned C++ submodule (ql/**/*.hpp at v1.42.1 @ 099987f0).
Match key  = class/struct NAME (QuantLib → pquantlib preserves PascalCase class names;
             only module paths are snake_cased). A C++ `class ClaytonCopula` is "ported"
             iff some Python file declares `class ClaytonCopula`.

This OVER-reports missing (counts nested helper structs, template tag types, detail::
helpers, forward-decls that slipped the filter) and UNDER-reports when Python renames a
class. It is a proxy, not gospel — but it is reproducible and it is the denominator we
drive to zero. Run from repo root.
"""

from __future__ import annotations

import collections
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
CPP = REPO / "migration-harness" / "cpp" / "quantlib" / "ql"
PY_ROOTS = [
    REPO / "pquantlib" / "src",
    REPO / "pquantlib-contrib" / "src",
    REPO / "pquantlib-helpers" / "src",
]

# class/struct Name, not a forward-decl (no trailing ';' immediately after the name),
# optionally preceded by a template<...> on the same logical line is fine — we just
# scan line-anchored declarations.
CPP_DECL = re.compile(r"^\s*(?:class|struct)\s+([A-Z][A-Za-z0-9_]*)\s*(?:[:{]|$)", re.MULTILINE)
PY_DECL = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


def cpp_subsystem(hpp: pathlib.Path) -> str:
    rel = hpp.relative_to(CPP).parts
    return rel[0] if len(rel) > 1 else "(root)"


def main() -> None:
    cpp: dict[str, str] = {}
    cpp_files = 0
    for hpp in CPP.rglob("*.hpp"):
        if hpp.name == "all.hpp":
            continue
        cpp_files += 1
        sub = cpp_subsystem(hpp)
        for name in CPP_DECL.findall(hpp.read_text(errors="ignore")):
            cpp.setdefault(name, sub)

    py: set[str] = set()
    for root in PY_ROOTS:
        if not root.exists():
            continue
        for pyf in root.rglob("*.py"):
            py.update(PY_DECL.findall(pyf.read_text(errors="ignore")))

    missing = {n: s for n, s in cpp.items() if n not in py}
    present = {n: s for n, s in cpp.items() if n in py}

    total_by_sub = collections.Counter(cpp.values())
    miss_by_sub = collections.Counter(missing.values())

    print(f"C++ headers scanned (excl all.hpp): {cpp_files}")
    print(f"C++ distinct class/struct names:    {len(cpp)}")
    print(f"  ported (name found in Python):    {len(present)}")
    print(f"  MISSING:                          {len(missing)}")
    print(f"  coverage:                         {100 * len(present) / len(cpp):.1f}%")
    print()
    print(f"{'subsystem':<22}{'total':>7}{'ported':>8}{'missing':>9}{'cov%':>7}")
    print("-" * 53)
    for sub in sorted(total_by_sub, key=lambda s: -miss_by_sub[s]):
        tot = total_by_sub[sub]
        miss = miss_by_sub[sub]
        cov = 100 * (tot - miss) / tot if tot else 100.0
        print(f"{sub:<22}{tot:>7}{tot - miss:>8}{miss:>9}{cov:>6.0f}%")

    # dump full missing list to CSV for tracking
    out = REPO / "migration-harness" / "coverage-gaps.csv"
    with out.open("w") as f:
        f.write("subsystem,class\n")
        for n, s in sorted(missing.items(), key=lambda kv: (kv[1], kv[0])):
            f.write(f"{s},{n}\n")
    print()
    print(f"full missing list → {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
