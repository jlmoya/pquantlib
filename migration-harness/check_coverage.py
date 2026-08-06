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
# A class declaration the gate must see. Three forms this deliberately covers,
# each of which the original pattern silently dropped from the DENOMINATOR —
# which is the dangerous direction of error: a name the gate never sees can
# never be reported missing, so the gate reads 0 while the class is absent.
#
#   template <class Curve> class GlobalBootstrap final : ...   one-line template
#   class Foo final : ...                                       'final'
#   class QL_DEPRECATED Foo : ...                                leading macro
#
# Forward declarations (`class Foo;`) stay excluded: the trailing group requires
# ':' or '{' or end-of-line, and ';' is neither.
CPP_DECL = re.compile(
    r"^\s*(?:template\s*<[^>\n]*>\s*)?"
    r"(?:class|struct)\s+"
    r"(?:[A-Z_][A-Z0-9_]*\s+)?"
    r"([A-Z][A-Za-z0-9_]*)"
    r"(?:\s+final)?\s*(?:[:{]|$)",
    re.MULTILINE,
)
PY_DECL = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


def cpp_subsystem(hpp: pathlib.Path) -> str:
    rel = hpp.relative_to(CPP).parts
    return rel[0] if len(rel) > 1 else "(root)"


# ---------------------------------------------------------------------------
# ALLOWLIST — reviewed exclusions.
#
# Mirrors JQuantLib's gate so that "done" means the same thing in both ports:
# UNFLAGGED == 0, where every C++ class is either (a) ported, or (b) listed here
# with a one-line rationale.
#
# NOTHING is allowlisted silently. Adding an entry is a reviewed decision, and
# the rationale is the review. A name belongs here only if it is a verified
# false positive of the exact-name match — a C++ idiom with no Python
# counterpart (traits/policy structs folded into duck typing, template tag
# types, detail:: helpers, nested Impl classes), or a class Python has under a
# different name.
#
# "We haven't got to it yet" is NOT a rationale. Deferred work belongs in
# docs/carve-outs.md and still counts as UNFLAGGED here, because a coverage
# gate that can be silenced by intent is not a gate.
# ---------------------------------------------------------------------------
ALLOWLIST: dict[str, str] = {
    # --- nested comparator / proxy structs with no standalone Python counterpart ---
    "CaseInsensitiveCompare":
        "nested comparator inside C++ IndexManager; Python normalises with name.lower() "
        "(pinned by the IndexManager tests)",
    "BaseCurrencyProxy":
        "private assignment-proxy struct inside C++ Money, exists only so "
        "`settings.conversionType() = X` compiles; Python properties give that syntax natively",
    "ConversionTypeProxy":
        "private assignment-proxy struct inside C++ Money; see BaseCurrencyProxy",
}


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
    allowlisted = {n: s for n, s in missing.items() if n in ALLOWLIST}
    unflagged = {n: s for n, s in missing.items() if n not in ALLOWLIST}

    total_by_sub = collections.Counter(cpp.values())
    miss_by_sub = collections.Counter(unflagged.values())

    print(f"C++ headers scanned (excl all.hpp): {cpp_files}")
    print(f"C++ distinct class/struct names:    {len(cpp)}")
    print(f"  ported (name found in Python):    {len(present)}")
    print(f"  missing (exact-name):             {len(missing)}")
    print(f"    - allowlisted (reviewed):       {len(allowlisted)}")
    print(f"    - UNFLAGGED (must reach 0):     {len(unflagged)}")
    print(f"  coverage (ported+allowlisted):    "
          f"{100 * (len(present) + len(allowlisted)) / len(cpp):.1f}%")
    print()
    print(f"{'subsystem':<22}{'total':>7}{'unflagged':>11}")
    print("-" * 41)
    if not unflagged:
        print("(no subsystem has unflagged gaps)")
    for sub in sorted(total_by_sub, key=lambda s: -miss_by_sub[s]):
        if not miss_by_sub[sub]:
            continue
        print(f"{sub:<22}{total_by_sub[sub]:>7}{miss_by_sub[sub]:>11}")

    # dump full missing list to CSV for tracking
    out = REPO / "migration-harness" / "coverage-gaps.csv"
    with out.open("w") as f:
        f.write("subsystem,class,status,rationale\n")
        for n, sub in sorted(allowlisted.items(), key=lambda kv: (kv[1], kv[0])):
            f.write(f'{sub},{n},ALLOWLISTED,"{ALLOWLIST[n]}"\n')
        for n, sub in sorted(unflagged.items(), key=lambda kv: (kv[1], kv[0])):
            f.write(f"{sub},{n},UNFLAGGED,\n")
    print()
    print(f"full gap list (allowlisted + unflagged) → {out.relative_to(REPO)}")
    if not unflagged:
        print()
        print("DONE: 0 unflagged gaps. "
              "Every C++ class is ported or allowlisted-with-rationale.")


if __name__ == "__main__":
    main()
