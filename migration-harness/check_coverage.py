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

    # --- C++ template / language idioms with no Python expression -------------
    "CuriouslyRecurringTemplate":
        "`template <class Impl> class CuriouslyRecurringTemplate` "
        "(ql/patterns/curiouslyrecurring.hpp:39) — a CRTP static-dispatch helper whose "
        "entire body is `static_cast<Impl&>(*this)`; Python resolves self.method() on the "
        "runtime class, so the idiom has no expression",
    "Foo":
        "not a declaration: a documentation placeholder inside the `\\code ... \\endcode` "
        "example blocks of ql/patterns/singleton.hpp:36 and ql/math/solver1d.hpp:43. The "
        "scanner reports it because it does not strip comments",
    "AcyclicVisitor":
        "`class AcyclicVisitor` (ql/patterns/visitor.hpp:33) is a DEGENERATE empty base — "
        "its only member is a virtual destructor — existing so C++ can dynamic_cast a "
        "visitor to Visitor<T>*; pquantlib.patterns.visitor.Visitor is a runtime_checkable "
        "Protocol and Event.accept isinstance-checks it (pquantlib/event.py), which is the "
        "same check without the tag base",
    "Proxy":
        "private nested `Observer::Proxy` (ql/patterns/observable.hpp:330) holding a mutex "
        "plus an `active_` flag so a dying Observer can be detached mid-notification; "
        "pquantlib.patterns.observer.Observable stores observers in a weakref.WeakSet "
        "(observer.py:27), which gives the same lifetime guarantee without the proxy",
    "Null":
        "`template <class Type> class Null` (ql/utilities/null.hpp:59) — a type-parameterised "
        "sentinel that implicitly converts to numeric_limits<float>::max(); this port spells "
        "the same sentinel `None` (pquantlib/prices.py, cashflows/dividend.py) and, where a "
        "float is structurally required, the NULL_REAL constant "
        "(math/statistics/general_statistics.py:32)",
    "Clone":
        "`template <class T> class Clone` (ql/utilities/clone.hpp:40) — a deep-copying smart "
        "pointer giving polymorphic VALUE semantics to a unique_ptr; Python names are already "
        "reference handles, so the port calls each product's own clone() directly "
        "(models/marketmodels/products/composite_product.py:153)",
    "ObservableValue":
        "`template <class T> class ObservableValue` (ql/utilities/observablevalue.hpp:42) — a "
        "value wrapper whose operator= notifies. QuantLib instantiates it in exactly ONE place, "
        "Settings::DateProxy (settings.hpp:41); the Python counterpart is the "
        "ObservableSettings.evaluation_date property setter, which notifies on change "
        "(patterns/observable_settings.py)",
    "DateProxy":
        "private nested `Settings::DateProxy : public ObservableValue<Date>` (ql/settings.hpp:41) "
        "existing only so `Settings::instance().evaluationDate() = d` compiles and so an unset "
        "date reads as today; both are the ObservableSettings.evaluation_date property plus "
        "evaluation_date_or_today() (patterns/observable_settings.py)",
    "Handle":
        "`template <class T> class Handle` (ql/handle.hpp:41) — the observable smart-pointer "
        "indirection that lets a curve/quote be swapped under its holders. This port threads "
        "the pointed-to object DIRECTLY and re-registers observers explicitly; the decision is "
        "documented at cashflows/cms_coupon_pricer.py:16 and implemented in every "
        "setTermStructure equivalent (e.g. termstructures/yield_/bond_helper.py:73)",
    "RelinkableHandle":
        "`template <class T> class RelinkableHandle : public Handle<T>` (ql/handle.hpp:117) — "
        "adds linkTo() to Handle; see the Handle entry. Relinking is modelled by re-assigning "
        "the object and re-registering, e.g. "
        "termstructures/yield_/overnight_index_future_rate_helper.py:111",
    "Link":
        "private nested `Handle::Link : public Observable, public Observer` (ql/handle.hpp:43) — "
        "the shared observable cell a Handle points at; it has no meaning without Handle, "
        "see the Handle entry",

    # --- renamed to fit the language's own conventions ------------------------
    "Error":
        "`class Error : public std::exception` (ql/errors.hpp:39) — QuantLib's single "
        "exception type. Python spells it pquantlib/exceptions.py:9 LibraryException, "
        "deriving from RuntimeError so `except RuntimeError` and `except Exception` behave "
        "as a Python caller expects; every qassert.require/fail raises it",
    "Position":
        "`struct Position { enum Type { Long, Short }; }` (ql/position.hpp:32) — a pure enum "
        "holder, C++'s way of namespacing an enum before enum class. Flattened to "
        "pquantlib/position.py:19 PositionType, with the C++ integer values",
    "Protection":
        "`struct Protection { enum Side { Buyer, Seller }; }` (ql/default.hpp:32) — pure enum "
        "holder; flattened to instruments/credit_default_swap.py:58 ProtectionSide "
        "(re-exported by experimental/credit/synthetic_cdo.py:47)",

    # --- static-only namespace holders: the Python MODULE is the namespace ----
    "DateParser":
        "`class DateParser` (ql/utilities/dataparsers.hpp:43) has no state and no instances — "
        "every member is a `static Date`. The Python module IS that namespace: "
        "pquantlib.time.date_parser.parse_iso (date_parser.py:27) and .parse_formatted "
        "(date_parser.py:48)",
    "PeriodParser":
        "`class PeriodParser` (ql/utilities/dataparsers.hpp:36) has no state and no instances — "
        "its single member is a `static Period parse`. The Python module IS that namespace: "
        "pquantlib.time.period_parser.parse (period_parser.py:55)",
    "Tracing":
        "`QuantLib::detail::Tracing` (ql/utilities/tracing.hpp:35) — a detail::-namespace "
        "singleton that exists ONLY to back the QL_TRACE* preprocessor macros, and whose "
        "enable() QL_FAILs unless the library was compiled with QL_ENABLE_TRACING. Python has "
        "no preprocessor; the stdlib `logging` module is the counterpart",

    # --- enum-holder structs C++ uses as namespaces, flattened in Python ------
    "Average":
        "`struct Average { enum Type { Arithmetic, Geometric }; }` "
        "(ql/instruments/averagetype.hpp:34) — a pure enum holder, C++'s way of namespacing an "
        "enum before enum class. Flattened to instruments/average_type.py:17 AverageType",
    "Barrier":
        "`struct Barrier { enum Type { DownIn, UpIn, DownOut, UpOut }; }` "
        "(ql/instruments/barriertype.hpp:35) — pure enum holder; flattened to "
        "instruments/barrier_option.py:29 BarrierType",
    "DoubleBarrier":
        "`struct DoubleBarrier { enum Type { KnockIn, KnockOut, KIKO, KOKI }; }` "
        "(ql/instruments/doublebarriertype.hpp:33) — pure enum holder; flattened to "
        "instruments/double_barrier_option.py:32 DoubleBarrierType",
    "PartialBarrier":
        "`struct PartialBarrier { enum Range { Start, EndB1, EndB2 }; }` "
        "(ql/instruments/partialtimebarrieroption.hpp:36) — pure enum holder; flattened to "
        "experimental/barrieroption/partial_time_barrier_option.py:38 PartialBarrierRange",
    "Futures":
        "`struct Futures { enum Type { IMM, ASX, Custom }; }` (ql/instruments/futures.hpp:33) — "
        "pure enum holder; flattened to termstructures/yield_/futures_rate_helper.py:35 "
        "FuturesType",
    "Settlement":
        "`struct Settlement` (ql/instruments/swaption.hpp:41) holds two enums plus one static "
        "validator. Flattened to instruments/swaption.py:54 SettlementType, :64 "
        "SettlementMethod and :76 check_settlement_type_and_method_consistency, which "
        "reproduces Settlement::checkTypeAndMethodConsistency (swaption.cpp:207-220)",
    "Price":
        "nested `class Bond::Price` (ql/instruments/bond.hpp:68) — amount + Dirty/Clean tag. "
        "Flattened to instruments/bond.py:71 BondPrice with instruments/bond.py:64 "
        "BondPriceType, since a module-scope `Price` would collide with the unrelated free "
        "`enum PriceType` of ql/prices.hpp",

    # --- inside a C++ comment block: not compiled, not shipped ----------------
    "StickyRatchetPayoff":
        "declared ONLY inside the /*--- ---*/ comment block at ql/instruments/stickyratchet.hpp"
        ":168-224, which C++ itself labels 'Old code for single sticky/ratchet payoffs, "
        "superated by DoubleStickyRatchetPayoff class above'. Not compiled; the live class is "
        "instruments/sticky_ratchet.py:39 DoubleStickyRatchetPayoff",
    "RatchetPayoff_2":
        "declared ONLY inside the dead /*--- ---*/ comment block of "
        "ql/instruments/stickyratchet.hpp (:197); see StickyRatchetPayoff",
    "StickyPayoff_2":
        "declared ONLY inside the dead /*--- ---*/ comment block of "
        "ql/instruments/stickyratchet.hpp (:213); see StickyRatchetPayoff",

    # --- flattened nested types Python carries under an owner-qualified name --
    "Impl":
        "the pimpl base nested in ~50 unrelated C++ types (ql/time/calendar.hpp:64, "
        "ql/time/daycounter.hpp:47, ql/models/parameter.hpp:41, ...), so one name cannot map to "
        "one Python class. Calendar/DayCounter drop the pimpl and are subclassed directly "
        "(time/calendar.py:34, daycounters/day_counter.py:25); Parameter keeps the strategy and "
        "names it models/parameter.py:58 ParameterImpl",
    "NumericalImpl":
        "nested `TermStructureFittingParameter::NumericalImpl` (ql/models/parameter.hpp:147); "
        "flattening it to module scope needs the owner in the name, so it is "
        "models/parameter.py:292 TermStructureFittingParameterImpl (which carries that C++ "
        "parity note in situ)",
    "ModelSettings":
        "nested `MarkovFunctional::ModelSettings` (ql/models/shortrate/onefactormodels/"
        "markovfunctional.hpp:115); flattened to module scope it needs the owner in the name, "
        "so it is models/shortrate/onefactor/markov_functional.py:105 MarkovFunctionalSettings",

    # --- hash-map key + hasher: Python tuples are hashable natively -----------
    "CachedSwapKey":
        "nested `Gaussian1dModel::CachedSwapKey` (ql/models/shortrate/onefactormodels/"
        "gaussian1dmodel.hpp:156) — a 3-field struct used only as an unordered_map key; the "
        "Python cache is keyed by a plain tuple, models/shortrate/gaussian1d_model.py:132",
    "CachedSwapKeyHasher":
        "nested `Gaussian1dModel::CachedSwapKeyHasher` (ql/models/shortrate/onefactormodels/"
        "gaussian1dmodel.hpp:166) — the std::hash functor CachedSwapKey needs; Python tuples "
        "are hashable without one, see CachedSwapKey",
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
