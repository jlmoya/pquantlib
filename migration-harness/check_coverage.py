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
# Block comments are stripped before scanning. Five C++ "classes" live only
# inside /* ... */ blocks — ESFIntegrator (saddlepointlossmodel.hpp:355, opened
# by "Just for testing ... not for release"), StickyRatchetPayoff,
# RatchetPayoff_2, StickyPayoff_2 (stickyratchet.hpp, under a header C++ itself
# labels "Old code ... superated by DoubleStickyRatchetPayoff") and Foo.
# They compile to nothing and cannot be instantiated, so counting them inflates
# the denominator with classes QuantLib does not have. Allowlisting each would
# also work but records them as real-but-excused, which they are not.
# Line comments need no handling: the pattern is line-anchored, so "// class X"
# cannot match.
CPP_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

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
#
# EVIDENCE STANDARD. "It is a nested class" is NOT sufficient on its own. An
# entry qualifies only when all three hold, and the rationale must say so:
#   1. the C++ declaration (file:line) shows it is nested / private / a tag
#      type / a detail:: template — i.e. not user-constructible on its own;
#   2. a named Python construct subsumes it; and
#   3. the BEHAVIOUR it carries is implemented AND cross-validated against
#      C++ in a named test module. If the behaviour is not tested, the name is
#      a real gap, not an allowlist candidate.
# ---------------------------------------------------------------------------
ALLOWLIST: dict[str, str] = {
    # =======================================================================
    # termstructures — C++ template/tag/nested idioms with no Python form.
    #
    # Every entry below states all three parts the evidence standard requires:
    # (1) the C++ declaration at file:line showing it is a tag type / policy
    # template / namespace-scoped enum holder / detail:: functor / protected
    # mixin; (2) the NAMED Python construct that subsumes it; (3) the NAMED
    # test module that cross-validates the behaviour it carries against C++.
    # =======================================================================

    # --- ZABR evaluation tag types -----------------------------------------
    # `struct ZabrShortMaturityLognormal {};` and its three siblings are EMPTY
    # structs (zabrsmilesection.hpp:42-45) whose only role is to select a
    # branch of `template <typename Evaluation> class ZabrSmileSection`
    # (hpp:47). They carry no state and no members. Python selects the same
    # branch with the ZabrEvaluation IntEnum
    # (math/interpolations/zabr_formula.py:81).
    "ZabrShortMaturityLognormal":
        "empty evaluation tag struct (ql/termstructures/volatility/zabrsmilesection.hpp:42), "
        "selecting a ZabrSmileSection<Evaluation> branch; Python uses "
        "ZabrEvaluation.ShortMaturityLognormal (math/interpolations/zabr_formula.py:81) and the "
        "branch's volatility/optionPrice are cross-validated against v1.43 in "
        "tests/termstructures/volatility/test_zabr_fd_smile_section.py "
        "(case 'short_maturity_lognormal')",
    "ZabrShortMaturityNormal":
        "empty evaluation tag struct (zabrsmilesection.hpp:43); Python uses "
        "ZabrEvaluation.ShortMaturityNormal (zabr_formula.py:81), cross-validated in "
        "tests/termstructures/volatility/test_zabr_fd_smile_section.py "
        "(case 'short_maturity_normal')",
    "ZabrLocalVolatility":
        "empty evaluation tag struct (zabrsmilesection.hpp:44); Python uses "
        "ZabrEvaluation.LocalVolatility (zabr_formula.py:81), cross-validated in "
        "tests/termstructures/volatility/test_zabr_fd_smile_section.py "
        "(cases 'local_volatility' and 'local_volatility_small_grid')",
    "ZabrFullFd":
        "empty evaluation tag struct (zabrsmilesection.hpp:45); Python uses "
        "ZabrEvaluation.FullFd (zabr_formula.py:81), cross-validated in "
        "tests/termstructures/volatility/test_zabr_fd_smile_section.py "
        "(case 'full_fd_small_grid')",

    # --- XABR cube model policy + typedef bundles --------------------------
    "XabrModelTraits":
        "`template <class Model> struct XabrModelTraits` "
        "(ql/termstructures/volatility/swaption/sabrswaptionvolatilitycube.hpp:79) — a pure "
        "static-policy template (nParams, createInterpolation, extractGamma, "
        "createSmileSection) with no state and no instances, specialised at "
        "zabrswaptionvolatilitycube.hpp:109 and noarbsabrswaptionvolatilitycube.hpp:46. Python "
        "dispatches the same three choices on the XabrModelKind IntEnum "
        "(termstructures/volatility/swaption/xabr_swaption_volatility_cube.py:68); nParams=4 vs 5, "
        "the interpolation factory and the smile-section factory are all cross-validated against "
        "v1.43 in tests/termstructures/volatility/swaption/test_xabr_cube_grid.py and "
        "test_xabr_swaption_volatility_cube.py",
    "SwaptionVolCubeSabrModel":
        "`struct SwaptionVolCubeSabrModel { typedef SABRInterpolation Interpolation; typedef "
        "SabrSmileSection SmileSection; }` (sabrswaptionvolatilitycube.hpp:1270) — a typedef "
        "bundle with no members, used only as the template argument of "
        "XabrSwaptionVolatilityCube. Python is XabrModelKind.SABR "
        "(xabr_swaption_volatility_cube.py:80), whose interpolation and smile-section choices are "
        "pinned by tests/termstructures/volatility/swaption/"
        "test_xabr_swaption_volatility_cube.py::test_xabr_cube_sabr_mode_smile_section_is_sabr_"
        "smile_section and ::test_xabr_cube_sabr_mode_matches_sabr_subclass_at_grid_pillar",
    "SwaptionVolCubeZabrModel":
        "`template <typename Kernel> struct SwaptionVolCubeZabrModel` "
        "(zabrswaptionvolatilitycube.hpp:99) — the same typedef bundle for ZabrInterpolation / "
        "ZabrSmileSection. Python is XabrModelKind.ZABR plus the zabr_evaluation argument "
        "(xabr_swaption_volatility_cube.py:81), pinned by tests/termstructures/volatility/"
        "swaption/test_xabr_swaption_volatility_cube.py::"
        "test_zabr_swaption_vol_cube_smile_section_is_zabr_smile_section and "
        "::test_zabr_swaption_vol_cube_xabr_parameters_returns_5tuple",

    # --- namespace-scoped enum holder --------------------------------------
    "Pillar":
        "`struct Pillar { enum Choice { MaturityDate, LastRelevantDate, CustomDate }; }` "
        "(ql/termstructures/bootstraphelper.hpp:40-47) — an enum holder with no members, "
        "existing only to namespace the enum. Python flattens it to the PillarChoice IntEnum "
        "(termstructures/bootstrap_helper.py:35), whose three values AND C++ ostream spellings "
        "are pinned by tests/termstructures/test_bootstrap_helper.py::"
        "test_pillar_choice_string_repr_matches_cpp",

    # --- detail:: comparator ----------------------------------------------
    "BootstrapHelperSorter":
        "`class BootstrapHelperSorter` inside `namespace detail` "
        "(ql/termstructures/bootstraphelper.hpp:248-258) — a comparator whose whole body is "
        "`h1->pillarDate() < h2->pillarDate()`, used only as the third argument of the std::sort "
        "in iterativebootstrap.hpp:163. Python is the sort key at "
        "termstructures/bootstrap/iterative_bootstrap.py:200 and local_bootstrap.py:169; its "
        "effect (a curve built from scrambled helpers equals the C++ curve node for node) is "
        "cross-validated against v1.43 in tests/termstructures/test_bootstrap_helper_sorter.py",

    # --- protected CRTP-style mixin ----------------------------------------
    "InterpolatedCurve":
        "`template <class Interpolator> class InterpolatedCurve` "
        "(ql/termstructures/interpolatedcurve.hpp:42) — every member except the destructor is "
        "PROTECTED (hpp:47 opens `protected:`); C++ says it exists so curves can 'use protected "
        "or private inheritance from this class to obtain the relevant data members'. It is not "
        "constructible on its own. Python holds times_/data_/dates_/interpolation directly on "
        "each concrete curve (termstructures/yield_/interpolated_discount_curve.py, "
        "interpolated_zero_curve.py, interpolated_forward_curve.py), and the mixin's whole "
        "observable contribution — the node grid feeding the interpolation — is cross-validated "
        "against v1.43 in tests/termstructures/yield_/test_interpolated_discount_curve.py"
        "::test_discount_at_nodes / ::test_discount_intermediate / ::test_discount_extrap and "
        "the matching zero/forward modules",

    # --- nested comparator / proxy structs with no standalone Python counterpart ---
    "CaseInsensitiveCompare":
        "nested comparator inside C++ IndexManager; Python normalises with name.lower() "
        "(pinned by the IndexManager tests)",
    "BaseCurrencyProxy":
        "private assignment-proxy struct inside C++ Money, exists only so "
        "`settings.conversionType() = X` compiles; Python properties give that syntax natively",
    "ConversionTypeProxy":
        "private assignment-proxy struct inside C++ Money; see BaseCurrencyProxy",

    # =======================================================================
    # time / day counters — private nested DayCounter::Impl subclasses.
    #
    # C++ hides the algorithm behind a pimpl chosen by the Convention enum;
    # Python's DayCounter classes dispatch on the same enum in one class. All
    # 9 Thirty360, 7 ActualActual and 3 Actual365Fixed conventions are
    # cross-validated against C++ probes, so every Impl below has its
    # behaviour pinned by name.
    # =======================================================================
    "Thirty360_Impl":
        "private nested base of the 30/360 family (thirty360.hpp:90); Python Thirty360 "
        "dispatches on Convention and all 9 are pinned in tests/daycounters/test_thirty_family.py",
    "US_Impl":
        "private nested DayCounter::Impl for Thirty360 USA (thirty360.hpp:97); "
        "Thirty360(Convention.USA), pinned in tests/daycounters/test_thirty_family.py",
    "EU_Impl":
        "private nested Impl for Thirty360 European/EurobondBasis (thirty360.hpp:107); "
        "Thirty360(Convention.European/.EurobondBasis), tests/daycounters/test_thirty_family.py",
    "IT_Impl":
        "private nested Impl for Thirty360 Italian (thirty360.hpp:112); "
        "Thirty360(Convention.Italian), pinned in tests/daycounters/test_thirty_family.py",
    "NASD_Impl":
        "private nested Impl for Thirty360 NASD (thirty360.hpp:126); "
        "Thirty360(Convention.NASD), pinned in tests/daycounters/test_thirty_family.py",
    "ISMA_Impl":
        "private nested Impl, two sites: Thirty360 ISMA/BondBasis (thirty360.hpp:102) and "
        "ActualActual ISMA/Bond WITH a Schedule (actualactual.hpp:57). Python: "
        "Thirty360(Convention.ISMA) and ActualActual(Convention.ISMA, schedule=...); pinned in "
        "tests/daycounters/test_thirty_family.py and test_actualactual.py::"
        "test_isma_with_schedule_matches_cpp",
    "ISDA_Impl":
        "private nested Impl, two sites: Thirty360 ISDA/German (thirty360.hpp:117) and "
        "ActualActual ISDA/Historical/Actual365 (actualactual.hpp:78). Python: "
        "Thirty360(Convention.ISDA) and ActualActual(Convention.ISDA); pinned in "
        "tests/daycounters/test_thirty_family.py and test_actualactual.py",
    "Old_ISMA_Impl":
        "private nested Impl for ActualActual ISMA/Bond WITHOUT a Schedule "
        "(actualactual.hpp:70); Python ActualActual(Convention.ISMA) with schedule=None, pinned in "
        "tests/daycounters/test_actualactual.py::test_isma_no_schedule_matches_cpp",
    "AFB_Impl":
        "private nested Impl for ActualActual AFB/Euro (actualactual.hpp:84); "
        "ActualActual(Convention.AFB/.Euro), pinned in tests/daycounters/test_actualactual.py::"
        "test_afb_family_matches_cpp",
    "CA_Impl":
        "private nested Impl for Actual365Fixed Canadian (actual365fixed.hpp:60); "
        "Actual365Fixed(Convention.Canadian), pinned in tests/daycounters/test_actual_family.py",
    "NL_Impl":
        "private nested Impl for Actual365Fixed NoLeap (actual365fixed.hpp:70); "
        "Actual365Fixed(Convention.NoLeap), pinned in tests/daycounters/test_actual_family.py",

    # =======================================================================
    # time / calendars — nested Calendar::Impl subclasses.
    #
    # C++ implements each market as a nested Impl selected by the Market enum;
    # Python has one class per calendar dispatching on a Market enum. Every
    # (calendar, market) pair below is cross-validated name + weekend mask +
    # every non-weekend holiday over 1901-2099 against C++ v1.43 in
    # pquantlib/tests/time/calendars/test_v143_calendar_markets.py, reference
    # migration-harness/references/v143/time/calmarkets.json.
    # =======================================================================
    "WesternImpl":
        "nested Calendar::Impl base fixing Sat+Sun weekends and the Gregorian easterMonday "
        "table (calendar.hpp:174); Python WesternCalendar (time/calendar.py:280), exercised by "
        "every Western calendar in tests/time/calendars/test_v143_calendar_markets.py",
    "OrthodoxImpl":
        "nested Calendar::Impl base fixing Sat+Sun weekends and the Julian easterMonday table "
        "(calendar.hpp:185); Python OrthodoxCalendar (time/calendar.py:292), exercised by the "
        "Romania/Russia/NorthMacedonia/Ukraine sections of test_v143_calendar_markets.py",
    "SettlementImpl":
        "nested Impl for the Settlement market of 12 calendars (australia.hpp:54, austria.hpp:74, "
        "brazil.hpp:80, canada.hpp:75, france.hpp:68, germany.hpp:114, italy.hpp:72, "
        "poland.hpp:53, russia.hpp:60, southkorea.hpp:79, unitedkingdom.hpp:93, "
        "unitedstates.hpp:158); Python <Calendar>(Market.Settlement), all 12 pinned in "
        "test_v143_calendar_markets.py",
    "ExchangeImpl":
        "nested Impl for the Exchange market of 6 calendars (austria.hpp:79, brazil.hpp:85, "
        "france.hpp:73, italy.hpp:77, russia.hpp:65 [MOEX], unitedkingdom.hpp:98); Python "
        "<Calendar>(Market.Exchange) / Russia(Market.MOEX), all 6 pinned in "
        "test_v143_calendar_markets.py",
    "MseImpl":
        "nested Impl at two sites: malta.hpp:66 and northmacedonia.hpp:53; Python "
        "Malta(MaltaMarket.MSE) and NorthMacedonia(NorthMacedoniaMarket.MSE), both pinned in "
        "test_v143_calendar_markets.py",
    "SseImpl":
        "nested Impl at two sites: china.hpp:60 and chile.hpp:62; Python China(Market.SSE) and "
        "Chile(Market.SSE), both pinned in test_v143_calendar_markets.py",
    "AsxImpl":
        "nested Impl (australia.hpp:59); Python Australia(Market.ASX), pinned in "
        "test_v143_calendar_markets.py",
    "TsxImpl":
        "nested Impl (canada.hpp:80); Python Canada(Market.TSX), pinned in "
        "test_v143_calendar_markets.py",
    "IbImpl":
        "nested Impl (china.hpp:67); Python China(Market.IB), pinned in "
        "test_v143_calendar_markets.py",
    "ZseImpl":
        "nested Impl (croatia.hpp:57); Python Croatia(CroatiaMarket.ZSE), pinned in "
        "test_v143_calendar_markets.py",
    "PseImpl":
        "nested Impl (czechrepublic.hpp:54); Python CzechRepublic(Market.PSE), pinned in "
        "test_v143_calendar_markets.py",
    "FrankfurtStockExchangeImpl":
        "nested Impl (germany.hpp:119); Python Germany(GermanyMarket.FrankfurtStockExchange), "
        "pinned in test_v143_calendar_markets.py",
    "XetraImpl":
        "nested Impl (germany.hpp:124); Python Germany(GermanyMarket.Xetra), pinned in "
        "test_v143_calendar_markets.py",
    "EurexImpl":
        "nested Impl (germany.hpp:129); Python Germany(GermanyMarket.Eurex), pinned in "
        "test_v143_calendar_markets.py",
    "EuwaxImpl":
        "nested Impl (germany.hpp:134); Python Germany(GermanyMarket.Euwax), pinned in "
        "test_v143_calendar_markets.py",
    "HkexImpl":
        "nested Impl (hongkong.hpp:65); Python HongKong(Market.HKEx), pinned in "
        "test_v143_calendar_markets.py",
    "IcexImpl":
        "nested Impl (iceland.hpp:55); Python Iceland(Market.ICEX), pinned in "
        "test_v143_calendar_markets.py",
    "NseImpl":
        "nested Impl (india.hpp:72); Python India(Market.NSE), pinned in "
        "test_v143_calendar_markets.py",
    "BejImpl":
        "nested Impl (indonesia.hpp:64), also selected by the JSX and IDX enumerators; Python "
        "Indonesia(Market.BEJ/.JSX/.IDX), all three pinned in test_v143_calendar_markets.py",
    "MervalImpl":
        "nested Impl (argentina.hpp:57); Python Argentina(Market.Merval), pinned in "
        "test_v143_calendar_markets.py",
    "BmvImpl":
        "nested Impl (mexico.hpp:59); Python Mexico(MexicoMarket.BMV), pinned in "
        "test_v143_calendar_markets.py",
    "MnseImpl":
        "nested Impl (montenegro.hpp:50); Python Montenegro(MontenegroMarket.MNSE), pinned in "
        "test_v143_calendar_markets.py",
    "CommonImpl":
        "nested Impl base shared by the two New Zealand markets (newzealand.hpp:68); Python "
        "NewZealand's shared rules, exercised by both the Wellington and Auckland sections of "
        "test_v143_calendar_markets.py",
    "WellingtonImpl":
        "nested Impl (newzealand.hpp:72); Python NewZealand(Market.Wellington), pinned in "
        "test_v143_calendar_markets.py",
    "AucklandImpl":
        "nested Impl (newzealand.hpp:77); Python NewZealand(Market.Auckland), pinned in "
        "test_v143_calendar_markets.py",
    "WseImpl":
        "nested Impl (poland.hpp:58); Python Poland(PolandMarket.WSE), pinned in "
        "test_v143_calendar_markets.py",
    "PublicImpl":
        "nested Impl (romania.hpp:61), both the Public market and the base of BVBImpl; Python "
        "Romania(RomaniaMarket.Public), pinned in test_v143_calendar_markets.py",
    "BVBImpl":
        "nested Impl (romania.hpp:66); Python Romania(RomaniaMarket.BVB), pinned in "
        "test_v143_calendar_markets.py",
    "TadawulImpl":
        "nested Impl (saudiarabia.hpp:52); Python SaudiArabia(SaudiArabiaMarket.Tadawul), pinned "
        "in test_v143_calendar_markets.py",
    "BseImpl":
        "nested Impl (serbia.hpp:50); Python Serbia(SerbiaMarket.BSE), pinned in "
        "test_v143_calendar_markets.py",
    "SgxImpl":
        "nested Impl (singapore.hpp:62); Python Singapore(SingaporeMarket.SGX), pinned in "
        "test_v143_calendar_markets.py",
    "BsseImpl":
        "nested Impl (slovakia.hpp:58); Python Slovakia(SlovakiaMarket.BSSE), pinned in "
        "test_v143_calendar_markets.py",
    "LseImpl":
        "nested Impl (slovenia.hpp:56); Python Slovenia(SloveniaMarket.LSE), pinned in "
        "test_v143_calendar_markets.py",
    "KrxImpl":
        "nested Impl (southkorea.hpp:85); Python SouthKorea(Market.KRX), pinned in "
        "test_v143_calendar_markets.py",
    "TsecImpl":
        "nested Impl (taiwan.hpp:58); Python Taiwan(TaiwanMarket.TSEC), pinned in "
        "test_v143_calendar_markets.py",
    "SetImpl":
        "nested Impl (thailand.hpp:71), the only impl of the single-market Thailand calendar; "
        "Python Thailand(), pinned in test_v143_calendar_markets.py",
    "UseImpl":
        "nested Impl (ukraine.hpp:55); Python Ukraine(Market.USE), pinned in "
        "test_v143_calendar_markets.py",
    "MetalsImpl":
        "nested Impl (unitedkingdom.hpp:103); Python UnitedKingdom(Market.Metals), pinned in "
        "test_v143_calendar_markets.py",
    "NyseImpl":
        "nested Impl (unitedstates.hpp:168); Python UnitedStates(Market.NYSE), pinned in "
        "test_v143_calendar_markets.py — including the pre-1980 special closings, since the "
        "reference runs from 1901",
    "GovernmentBondImpl":
        "nested Impl (unitedstates.hpp:173), also the base of SofrImpl; Python "
        "UnitedStates(Market.GovernmentBond), pinned in test_v143_calendar_markets.py",
    "SofrImpl":
        "nested Impl (unitedstates.hpp:178); Python UnitedStates(Market.SOFR), pinned in "
        "test_v143_calendar_markets.py",
    "NercImpl":
        "nested Impl (unitedstates.hpp:183); Python UnitedStates(Market.NERC), pinned in "
        "test_v143_calendar_markets.py",
    "FederalReserveImpl":
        "nested Impl (unitedstates.hpp:190); Python UnitedStates(Market.FederalReserve), pinned "
        "in test_v143_calendar_markets.py",
    "LiborImpactImpl":
        "nested Impl (unitedstates.hpp:163); Python UnitedStates(Market.LiborImpact), pinned in "
        "test_v143_calendar_markets.py",

    # =======================================================================
    # math — Array / Matrix.
    # =======================================================================
    "Array":
        "hand-rolled rank-1 container (array.hpp:52); pquantlib/math/array.py makes Array a type "
        "ALIAS for numpy.typing.NDArray[float64]. Every operation the C++ class provides "
        "(indexing, unary minus, elementwise and scalar arithmetic, DotProduct, Norm2, Abs, "
        "Sqrt, Log, Exp, Pow) is diffed against C++ v1.43 in tests/math/test_v143_array_matrix.py",
    "Matrix":
        "hand-rolled rank-2 container (matrix.hpp:41); pquantlib/math/matrix.py makes Matrix a "
        "type ALIAS for numpy.typing.NDArray[float64]. Arithmetic, matrix/vector products, "
        "transpose, outerProduct, inverse, determinant and the row/column/diagonal views are "
        "diffed against C++ v1.43 in tests/math/test_v143_array_matrix.py; the decompositions "
        "(Cholesky, SVD, symmetric Schur, TQR, Householder) in tests/math/matrixutilities/",

    # =======================================================================
    # math / interpolations — nested Interpolation::Impl pimpls.
    #
    # C++ splits each interpolation into a user-facing class plus a nested
    # detail:: Impl holding the algorithm, because Interpolation is a
    # handle-body type parameterised on iterator types. Python has no such
    # split: one class carries both. Each entry names the Python class and the
    # cross-validated test that pins the algorithm.
    # =======================================================================
    "AbcdInterpolationImpl":
        "nested detail:: Impl (abcdinterpolation.hpp:78); Python AbcdInterpolation "
        "(math/interpolations/abcd_interpolation.py), pinned in "
        "tests/math/interpolations/test_abcd_interpolation.py",
    "BackwardFlatInterpolationImpl":
        "nested detail:: Impl (backwardflatinterpolation.hpp:71); Python "
        "BackwardFlatInterpolation (math/interpolations/backward_flat.py), pinned in "
        "tests/math/interpolations/test_interpolation_factories.py and test_interpolations.py",
    "BackwardflatLinearInterpolationImpl":
        "nested detail:: Impl (backwardflatlinearinterpolation.hpp:34); Python "
        "BackwardflatLinearInterpolation, pinned in "
        "tests/math/interpolations/test_backwardflat_linear_interpolation.py",
    "BicubicSplineImpl":
        "nested detail:: Impl (bicubicsplineinterpolation.hpp:46); Python BicubicSpline "
        "(math/interpolations/bicubic_spline.py), pinned in "
        "tests/math/interpolations/test_bicubic_spline.py and test_bicubic_spline_v143.py",
    "BilinearInterpolationImpl":
        "nested detail:: Impl (bilinearinterpolation.hpp:35); Python BilinearInterpolation "
        "(math/interpolations/bilinear.py), pinned in "
        "tests/math/interpolations/test_bilinear_v143.py",
    "ConvexMonotoneImpl":
        "nested detail:: Impl (convexmonotoneinterpolation.hpp:174); Python "
        "ConvexMonotoneInterpolation, pinned in "
        "tests/math/interpolations/test_convex_monotone_interpolation.py",
    "CubicInterpolationImpl":
        "nested detail:: Impl (cubicinterpolation.hpp:377); Python CubicInterpolation, whose "
        "coefficient-storage base CubicInterpolationBaseImpl mirrors C++ "
        "detail::CubicInterpolationBaseImpl; pinned in "
        "tests/math/interpolations/test_cubic_interpolation.py",
    "FlatExtrapolatorImpl":
        "nested detail:: Impl (flatextrapolation.hpp:48); Python FlatExtrapolator, pinned in "
        "tests/math/interpolations/test_flat_extrapolation.py",
    "FlatExtrapolator2DImpl":
        "nested detail:: Impl (flatextrapolation2d.hpp:43); Python FlatExtrapolator2D, pinned in "
        "tests/math/interpolations/test_flat_extrapolation_2d.py",
    "ForwardFlatInterpolationImpl":
        "nested detail:: Impl (forwardflatinterpolation.hpp:71); Python ForwardFlatInterpolation "
        "(math/interpolations/forward_flat.py), pinned in "
        "tests/math/interpolations/test_interpolation_factories.py and test_interpolations.py",
    "KernelInterpolationImpl":
        "nested detail:: Impl (kernelinterpolation.hpp:37); Python KernelInterpolation, pinned "
        "in tests/math/interpolations/test_kernel_interpolation.py",
    "KernelInterpolation2DImpl":
        "nested detail:: Impl (kernelinterpolation2d.hpp:56); Python KernelInterpolation2D, "
        "pinned in tests/math/interpolations/test_kernel_interpolation_2d.py",
    "LagrangeInterpolationImpl":
        "nested detail:: Impl (lagrangeinterpolation.hpp:43); Python LagrangeInterpolation, "
        "pinned in tests/math/test_lagrange_interpolation.py and "
        "tests/math/interpolations/test_lagrange_updated_y.py",
    "LinearInterpolationImpl":
        "nested detail:: Impl (linearinterpolation.hpp:72); Python LinearInterpolation "
        "(math/interpolations/linear.py), pinned in "
        "tests/math/interpolations/test_interpolation_factories.py and test_interpolations.py",
    "LogInterpolationImpl":
        "nested detail:: Impl (loginterpolation.hpp:366); Python LogInterpolation "
        "(math/interpolations/log_interpolation.py:70, which cites it as its parity target), "
        "pinned in tests/math/interpolations/test_log_interpolation.py",
    "MixedInterpolationImpl":
        "nested detail:: Impl (mixedinterpolation.hpp:224); Python "
        "MixedLinearCubicInterpolation, pinned in "
        "tests/math/interpolations/test_mixed_interpolation.py",

    # =======================================================================
    # math / XABR template machinery.
    # =======================================================================
    "XABRInterpolationImpl":
        "nested detail:: template combining Interpolation::templateImpl with XABRCoeffHolder "
        "(xabrinterpolation.hpp:101); Python folds it into each concrete fitter - "
        "SABRInterpolation, ZabrInterpolation, NoArbSabrInterpolation - each cross-validated in "
        "tests/math/interpolations/test_sabr_interpolation.py, test_zabr_interpolation.py and "
        "tests/experimental/volatility/test_no_arb_sabr_interpolation.py",
    "XABRCoeffHolder":
        "detail:: template mixin holding calibration state - params_, error_, maxError_, "
        "XABREndCriteria_ (xabrinterpolation.hpp:51); in Python that state lives directly on the "
        "fitter and is read through alpha()/beta()/nu()/rho()/gamma()/rms_error()/max_error(), "
        "asserted in tests/math/interpolations/test_sabr_interpolation.py and "
        "test_zabr_interpolation.py",
    "XABRError":
        "private nested CostFunction inside XABRInterpolationImpl (xabrinterpolation.hpp:288), "
        "reachable only during calibration; Python passes residuals straight to "
        "scipy.optimize.least_squares, and the resulting fits are cross-validated in "
        "tests/math/interpolations/test_sabr_interpolation.py and test_zabr_interpolation.py",

    # =======================================================================
    # math / MultiCubicSpline template scaffolding.
    #
    # C++ builds an arbitrary-rank tensor-product spline out of recursive
    # template types and tag structs. Python's MultiCubicSpline
    # (math/interpolations/multi_cubic_spline.py) uses n-d numpy arrays and
    # ordinary recursion, so none of the scaffolding has - or needs - a
    # counterpart. The resulting interpolant is cross-validated on uniform and
    # non-uniform 2-D and 3-D grids in
    # tests/math/interpolations/test_multi_cubic_spline.py.
    # =======================================================================
    "EmptyArg":
        "empty tag struct used as an arg_type recursion terminator "
        "(multicubicspline.hpp:44); no runtime meaning - see the MultiCubicSpline note above",
    "EmptyRes":
        "empty tag struct used as a res_type recursion terminator "
        "(multicubicspline.hpp:45); no runtime meaning - see the MultiCubicSpline note above",
    "EmptyDim":
        "empty tag struct used as a size_t recursion terminator "
        "(multicubicspline.hpp:46); no runtime meaning - see the MultiCubicSpline note above",
    "DataTable":
        "recursive template holding one nesting level of the n-d coefficient table "
        "(multicubicspline.hpp:48); Python uses a single n-d numpy array in MultiCubicSpline",
    "Data":
        "recursive head/tail template pair for the n-d value grid (multicubicspline.hpp:89), "
        "plus three unrelated PROTECTED nested POD holders in experimental/commodities "
        "(paymentterm.hpp:57, unitofmeasureconversion.hpp:85, commoditytype.hpp:65). Those three "
        "are ported as the module-private _Data dataclasses in commodity_type.py:26, "
        "payment_term.py:29 and unit_of_measure_conversion.py:31 - same shared-body role, "
        "including the interned registry - and are exercised by "
        "tests/experimental/commodities/test_commodity_type.py, test_unit_of_measure_conversion.py "
        "and test_commodity_base_and_periods.py",
    "Point":
        "recursive head/tail template pair for an n-d evaluation point "
        "(multicubicspline.hpp:122); Python passes a length-n sequence of coordinates to "
        "MultiCubicSpline.__call__ (multi_cubic_spline.py:147)",
    "Int2Type":
        "compile-time integer-to-type map selecting the rank-specific spline/splint typedefs "
        "(multicubicspline.hpp:372); Python dispatches on len(grid) at runtime",

    # --- merged from wave 10 (models/legacy/processes/root/quotes/utilities/patterns) ---
    "CuriouslyRecurringTemplate":
        "`template <class Impl> class CuriouslyRecurringTemplate` "
        "(ql/patterns/curiouslyrecurring.hpp:39) — a CRTP static-dispatch helper whose "
        "entire body is `static_cast<Impl&>(*this)`; Python resolves self.method() on the "
        "runtime class, so the idiom has no expression",
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
        for name in CPP_DECL.findall(CPP_BLOCK_COMMENT.sub("", hpp.read_text(errors="ignore"))):
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
