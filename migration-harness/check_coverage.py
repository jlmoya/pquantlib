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
