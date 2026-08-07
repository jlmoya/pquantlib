"""MultipleResetsSwapRateHelper, cross-validated against C++ v1.43.

Reference: ``migration-harness/references/v143/ts/ratehelpers.json``
Probe:     ``migration-harness/cpp/probes/v143_ts_ratehelpers/probe.cpp``

22 configurations pin the helper's dates and its implied quote. Every
constructor argument past the required five gets at least one non-default
case: ``discounting_curve``, ``averaging_method``, ``spread``,
``fixed_frequency``, ``fixed_day_count`` and ``fixed_convention``.

Two things these cases are here to hold down:

* ``latest_date`` is ``max(fixed_leg[-1].date(), floating_leg[-1].date())`` —
  the legs' final PAYMENT dates (multipleresetsswaphelper.cpp:70-71). The
  ``mrs_fixed_following`` / ``mrs_fixed_preceding`` pair rolls the fixed leg on
  a different convention from the floating one, which is what lets one leg win
  the max on its own.
* the helper has no ``Pillar::Choice``: ``maturity_date()`` falls through to
  ``latest_relevant_date()`` and ``pillar_date()`` to ``latest_date()``
  (bootstraphelper.hpp:179-203). :func:`test_helper_dates` asserts the
  fall-through rather than assuming it.

Tolerance: EXACT for dates and counts; TIGHT for the implied quote, the NPV
and the fixed-leg BPS. ``fair_rate`` is ``-(floatingLegNPV + spreadNPV) /
(fixedLegBPS / 1e-4)`` — discounted sums and one division, no solver.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.index_manager import IndexManager
from pquantlib.indexes.interest_rate_index import InterestRateIndex
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.multiple_resets_swap_rate_helper import (
    MultipleResetsSwapRateHelper,
)
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_YEARS = TimeUnit.Years
_MONTHS = TimeUnit.Months

#: probe.cpp:465 — the single evaluation date for this block.
_EVAL = Date.from_ymd(15, Month.June, 2026)
#: probe.cpp — the three flat curves.
_FORECAST_RATE = 0.030
_DISCOUNT_RATE = 0.025
_BOOTSTRAPPED_RATE = 0.028
#: probe.cpp — the seeding window, in calendar days before the eval date.
_SEED_DAYS_BACK = 30

_MF = BusinessDayConvention.ModifiedFollowing


@dataclass(frozen=True)
class _Case:
    """One row of the probe's MRS ``cases`` vector (probe.cpp:534-575)."""

    tenor: Period = field(default_factory=lambda: Period(1, _YEARS))
    settlement_days: int = 2
    index_tenor: Period = field(default_factory=lambda: Period(3, _MONTHS))
    resets_per_coupon: int = 2
    with_discount_curve: bool = False
    averaging: RateAveraging = RateAveraging.Compound
    spread: float = 0.0
    fixed_frequency: Frequency = Frequency.NoFrequency
    fixed_day_count: DayCounter | None = None
    fixed_convention: BusinessDayConvention = _MF
    quote: float = 0.025


_1Y = Period(1, _YEARS)
_3M = Period(3, _MONTHS)

CASES: dict[str, _Case] = {
    "mrs_base": _Case(),
    # Tenor sweep. 2Y and 3Y off the 17-Jun-2026 spot end on a Saturday and a
    # Sunday; MakeMultipleResetsSwap adjusts the end date BEFORE generating the
    # reset schedule, leaving 9 / 13 reset periods — see
    # :func:`test_odd_reset_count_is_rejected`.
    "mrs_4y": _Case(tenor=Period(4, _YEARS)),
    "mrs_5y": _Case(tenor=Period(5, _YEARS)),
    "mrs_10y": _Case(tenor=Period(10, _YEARS)),
    # resets_per_coupon.
    "mrs_resets4": _Case(resets_per_coupon=4),
    "mrs_5y_resets4": _Case(tenor=Period(5, _YEARS), resets_per_coupon=4),
    "mrs_1m_index_resets3": _Case(index_tenor=Period(1, _MONTHS), resets_per_coupon=3),
    "mrs_6m_index_resets2": _Case(tenor=Period(5, _YEARS), index_tenor=Period(6, _MONTHS)),
    # settlement_days.
    "mrs_settle0": _Case(settlement_days=0),
    "mrs_settle5": _Case(settlement_days=5),
    # discounting_curve.
    "mrs_discount_curve": _Case(with_discount_curve=True),
    "mrs_5y_discount_curve": _Case(tenor=Period(5, _YEARS), with_discount_curve=True),
    # averaging_method.
    "mrs_simple": _Case(averaging=RateAveraging.Simple),
    "mrs_5y_simple": _Case(tenor=Period(5, _YEARS), averaging=RateAveraging.Simple),
    # spread.
    "mrs_spread": _Case(spread=0.0025),
    "mrs_negative_spread": _Case(spread=-0.0015),
    # fixed-leg overrides.
    "mrs_fixed_annual": _Case(tenor=Period(5, _YEARS), fixed_frequency=Frequency.Annual),
    "mrs_fixed_daycount": _Case(fixed_day_count=Actual365Fixed()),
    "mrs_fixed_thirty360": _Case(
        tenor=Period(5, _YEARS), fixed_day_count=Thirty360(Thirty360Convention.BondBasis)
    ),
    "mrs_fixed_following": _Case(
        tenor=Period(5, _YEARS), fixed_convention=BusinessDayConvention.Following
    ),
    "mrs_fixed_preceding": _Case(
        tenor=Period(5, _YEARS), fixed_convention=BusinessDayConvention.Preceding
    ),
    # The quote only moves quote_error, never the dates or the implied quote.
    "mrs_quote_04": _Case(quote=0.04),
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/ts/ratehelpers")


@pytest.fixture(autouse=True)
def _global_state() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Pin the evaluation date and restore it, plus the fixing histories.

    The probe sets ``Settings::instance().evaluationDate() = kMrsEval`` and
    clears the histories per case (probe.cpp:479-480); both are global, so both
    are undone here.
    """
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = _EVAL  # probe.cpp:479
    IndexManager().clear_histories()
    try:
        yield
    finally:
        settings.evaluation_date = previous
        IndexManager().clear_histories()


def _flat(rate: float) -> YieldTermStructureProtocol:
    return cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            _EVAL, rate, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
        ),
    )


def _seed_fixings(index: InterestRateIndex) -> None:
    """probe.cpp:155-161 / 483 — the ``settlement_days == 0`` case fixes in the past."""
    d = _EVAL - _SEED_DAYS_BACK
    while d < _EVAL:
        if index.is_valid_fixing_date(d):
            index.add_fixing(d, 0.02 + 1.0e-5 * float(d.serial_number() % 100), True)
        d = d + 1


def _build(case: _Case) -> MultipleResetsSwapRateHelper:
    """Mirror ``emitMrs`` (probe.cpp:478-500)."""
    IndexManager().clear_histories()
    index = Euribor(case.index_tenor, _flat(_FORECAST_RATE))
    _seed_fixings(index)

    helper = MultipleResetsSwapRateHelper(
        case.settlement_days,
        case.tenor,
        case.quote,
        index,
        case.resets_per_coupon,
        _flat(_DISCOUNT_RATE) if case.with_discount_curve else None,
        case.averaging,
        case.spread,
        case.fixed_frequency,
        case.fixed_day_count,
        case.fixed_convention,
    )
    helper.set_term_structure(_flat(_BOOTSTRAPPED_RATE))
    return helper


def _iso(d: Date) -> str:
    return f"{d.year()}-{int(d.month()):02d}-{d.day_of_month():02d}"


def _assert_date(actual: Date, ref: dict[str, Any], name: str) -> None:
    assert actual.serial_number() == ref[f"{name}_serial"], f"{name}: serial"
    assert _iso(actual) == ref[f"{name}_iso"], f"{name}: ISO"


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(CASES))
def test_helper_dates(key: str, cpp: dict[str, Any]) -> None:
    """The five date accessors and the swap they come from. EXACT tier.

    ``maturity_date()`` and ``pillar_date()`` are the fall-through values, not
    values C++ ever assigns — asserting them keeps a port from inventing a
    pillar choice this helper does not have.
    """
    ref = cpp[key]
    helper = _build(CASES[key])
    _assert_date(helper.earliest_date(), ref, "earliest")
    _assert_date(helper.latest_date(), ref, "latest")
    _assert_date(helper.maturity_date(), ref, "maturity")
    _assert_date(helper.latest_relevant_date(), ref, "latest_relevant")
    _assert_date(helper.pillar_date(), ref, "pillar")

    swap = helper.swap()
    _assert_date(swap.start_date(), ref, "swap_start")
    _assert_date(swap.maturity_date(), ref, "swap_maturity")
    _assert_date(swap.fixed_leg()[-1].date(), ref, "fixed_leg_last_payment")
    _assert_date(swap.floating_leg()[-1].date(), ref, "floating_leg_last_payment")
    assert len(swap.fixed_leg()) == ref["n_fixed_coupons"]
    assert len(swap.floating_leg()) == ref["n_floating_coupons"]


@pytest.mark.parametrize("key", list(CASES))
def test_latest_date_is_the_max_of_the_two_legs(key: str, cpp: dict[str, Any]) -> None:
    """multipleresetsswaphelper.cpp:70-71, restated as an identity on the reference."""
    ref = cpp[key]
    assert ref["latest_serial"] == max(
        ref["fixed_leg_last_payment_serial"], ref["floating_leg_last_payment_serial"]
    )
    helper = _build(CASES[key])
    swap = helper.swap()
    assert helper.latest_date() == max(
        swap.fixed_leg()[-1].date(), swap.floating_leg()[-1].date()
    )
    # earliestDate_ is the swap start, not the evaluation date or the spot.
    assert helper.earliest_date() == swap.start_date()


def test_fixed_leg_convention_reaches_the_dates(cpp: dict[str, Any]) -> None:
    """``fixed_convention`` is not decorative: Preceding moves a payment date.

    The 5Y schedules end on 17 June 2031, a TARGET business day, so the
    terminal date is untouched; the difference shows up on the interior fixed
    coupons and therefore in the implied quote.
    """
    following = cpp["mrs_fixed_following"]
    preceding = cpp["mrs_fixed_preceding"]
    assert following["implied_quote"] != preceding["implied_quote"]
    tight(_build(CASES["mrs_fixed_following"]).implied_quote(), following["implied_quote"])
    tight(_build(CASES["mrs_fixed_preceding"]).implied_quote(), preceding["implied_quote"])


def test_dates_re_initialize_when_the_evaluation_date_moves() -> None:
    """This is a ``RelativeDateRateHelper``: the dates follow the evaluation date.

    The probe runs the whole block at one evaluation date, so there is no C++
    row to compare a moved helper against; what is asserted instead is the
    contract itself — a helper built at ``_EVAL`` and then re-pointed at a
    later date must land on exactly the dates a helper freshly built there
    gets. A port that computed the dates once in ``__init__`` keeps the old
    ones and fails both halves.
    """
    moved_to = _EVAL + 7
    helper = _build(_Case())
    before = helper.earliest_date()

    ObservableSettings().evaluation_date = moved_to
    fresh = _build(_Case())
    assert helper.earliest_date() != before
    assert helper.earliest_date() == fresh.earliest_date()
    assert helper.latest_date() == fresh.latest_date()
    assert helper.pillar_date() == fresh.pillar_date()


def test_odd_reset_count_is_rejected() -> None:
    """The 2Y tenor's adjusted end leaves 9 reset periods for 2 resets/coupon.

    C++ builds the reset schedule from the ALREADY-adjusted end date
    (makemultipleresetsswap.cpp), so a tenor whose unadjusted end lands on a
    weekend seeds the backward roll one day off the start.
    ``MultipleResetsSwap`` then rejects the odd count — which is why the tenor
    sweep above skips 2Y and 3Y rather than quietly using them.
    """
    index = Euribor(_3M, _flat(_FORECAST_RATE))
    with pytest.raises(LibraryException, match="not a multiple of"):
        MultipleResetsSwapRateHelper(2, Period(2, _YEARS), 0.025, index, 2)


# ---------------------------------------------------------------------------
# Implied quote
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(CASES))
def test_implied_quote(key: str, cpp: dict[str, Any]) -> None:
    """``swap.fair_rate()`` plus the NPV and BPS behind it. TIGHT tier.

    The cloned index forecasts off the curve passed to ``set_term_structure``
    (2.8%) while discounting is either that curve or the exogenous one (2.5%),
    so the ``mrs_*_discount_curve`` cases separate a port that ignores
    ``discounting_curve``.
    """
    ref = cpp[key]
    helper = _build(CASES[key])
    tight(helper.implied_quote(), ref["implied_quote"], reason=f"{key} implied quote")
    tight(helper.quote_error(), ref["quote_error"], reason=f"{key} quote error")

    swap = helper.swap()
    tight(swap.npv(), ref["swap_npv"], reason=f"{key} swap NPV")
    tight(swap.fixed_leg_bps(), ref["fixed_leg_bps"], reason=f"{key} fixed-leg BPS")
    tight(
        swap.floating_leg_npv(), ref["floating_leg_npv"], reason=f"{key} floating-leg NPV"
    )


@pytest.mark.parametrize("key", list(CASES))
def test_swap_carries_the_constructor_arguments(key: str, cpp: dict[str, Any]) -> None:
    """The knobs reach the instrument, not just the helper. EXACT / TIGHT."""
    ref = cpp[key]
    case = CASES[key]
    helper = _build(case)
    swap = helper.swap()
    assert swap.resets_per_coupon() == ref["resets_per_coupon"]
    assert int(swap.averaging_method()) == ref["averaging_method"]
    tight(swap.spread(), ref["spread"], reason=f"{key} spread")
    # C++ builds the swap with a fixed rate of 0.0 and reads fairRate off it
    # (multipleresetsswaphelper.cpp:58).
    exact(swap.fixed_rate(), ref["fixed_rate"])
    exact(swap.fixed_rate(), 0.0)
    # ...and the helper's own inspectors agree with the case.
    assert helper.resets_per_coupon() == case.resets_per_coupon
    assert helper.averaging_method() == case.averaging
    exact(helper.spread(), case.spread)


def test_discount_curve_changes_the_answer(cpp: dict[str, Any]) -> None:
    """Supplying ``discounting_curve`` must move the implied quote.

    Without it C++ links the discount handle to the curve being bootstrapped
    (multipleresetsswaphelper.cpp:78-81); with it, to the exogenous one.
    """
    assert cpp["mrs_5y"]["implied_quote"] != cpp["mrs_5y_discount_curve"]["implied_quote"]
    assert _build(CASES["mrs_5y"]).implied_quote() != _build(
        CASES["mrs_5y_discount_curve"]
    ).implied_quote()


def test_empty_fixed_day_count_falls_back_to_the_index(cpp: dict[str, Any]) -> None:
    """multipleresetsswaphelper.cpp:42 — an empty DayCounter means the index's.

    Euribor3M is Actual/360, so passing ``None`` must agree with passing
    ``Actual360()`` explicitly and differ from ``Actual365Fixed()``.
    """
    default = _build(CASES["mrs_base"])
    explicit = _build(_Case(fixed_day_count=Actual360()))
    tight(explicit.implied_quote(), default.implied_quote())
    tight(default.implied_quote(), cpp["mrs_base"]["implied_quote"])
    assert cpp["mrs_fixed_daycount"]["implied_quote"] != cpp["mrs_base"]["implied_quote"]


def test_implied_quote_requires_a_term_structure() -> None:
    """C++ ``QL_REQUIRE(termStructure_ != nullptr, ...)``
    (multipleresetsswaphelper.cpp:86)."""
    index = Euribor(_3M, _flat(_FORECAST_RATE))
    helper = MultipleResetsSwapRateHelper(2, _1Y, 0.025, index, 2)
    with pytest.raises(LibraryException, match="term structure not set"):
        helper.implied_quote()
