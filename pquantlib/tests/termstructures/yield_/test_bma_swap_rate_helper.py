"""BMASwapRateHelper dates + implied quote, cross-validated against C++ v1.43.

Reference: ``migration-harness/references/v143/ts/ratehelpers.json``
Probe:     ``migration-harness/cpp/probes/v143_ts_ratehelpers/probe.cpp``

30 configurations pin the two dates the helper actually computes
(``earliest_date`` and ``latest_date``) plus the three that fall through to
``latest_date``, the underlying swap's schedules, and ``implied_quote``.

The two hard bits, and the cases that catch them:

* ``earliest_date`` rolls the evaluation date on the JOINT calendar of the
  helper calendar and the ibor index's fixing calendar (ratehelpers.cpp:678-680),
  then advances ``settlement_days`` on the helper calendar ALONE
  (ratehelpers.cpp:681-682). ``bma_uk_cal_5y_us_holiday`` separates the two: on
  4 July 2024 the US is shut and the UK is not, so a joint calendar built from
  two UK calendars leaves the reference date on the 4th while the
  US-containing joint calendar rolls it to the 5th — one business day of
  difference that propagates into every date and into the implied quote.
* ``latest_date`` is the value date of the Wednesday AFTER the adjusted swap
  maturity (ratehelpers.cpp:716-722), not the maturity itself. Because the
  branch is ``w >= 4``, a maturity that already falls on a Wednesday rolls a
  further full week: ``bma_5y_saturday`` (maturity Wed 24 Jan 2029, latest
  1 Feb) is that case. ``bma_target_cal_5y_target_holiday_maturity`` versus
  ``bma_joint_cal_5y_target_holiday_maturity`` is the sharpest pair — same
  quote, same tenor, different helper calendar, and ``latest_date`` comes out
  a week apart (10 May vs 3 May 2029).

Tolerance: EXACT for every date, weekday and count; TIGHT for the implied
quote and the NPVs behind it. There is no solver in the path — the quote is
``-0.75 * bmaLegNPV / liborLegNPV``, i.e. two discounted cashflow sums and one
division.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

import pytest

from pquantlib.cashflows.coupon import Coupon
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.actual_actual import ActualActual
from pquantlib.daycounters.actual_actual import Convention as AAConvention
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.bma_index import BMAIndex
from pquantlib.indexes.ibor.usd_libor import USDLibor
from pquantlib.indexes.index_manager import IndexManager
from pquantlib.indexes.interest_rate_index import InterestRateIndex
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.bma_swap_rate_helper import BMASwapRateHelper
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.joint_calendar import JointCalendar
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.calendars.united_states import UnitedStates
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_YEARS = TimeUnit.Years
_MONTHS = TimeUnit.Months

#: probe.cpp:196-197 — the quoted Libor fraction.
_QUOTE = 0.70
#: probe.cpp:198 — the curve being bootstrapped (the BMA leg forecasts off it).
_BOOTSTRAPPED_RATE = 0.020
#: probe.cpp:199 — the ibor index's own curve (Libor leg + discounting).
_LIBOR_CURVE_RATE = 0.035
#: probe.cpp:222 — the seeding window, in calendar days before the eval date.
_SEED_DAYS_BACK = 60

_Q = Period(3, _MONTHS)
_AA_ISDA = ActualActual(AAConvention.ISDA)
_FOLLOWING = BusinessDayConvention.Following


def _us() -> Calendar:
    return UnitedStates(UnitedStates.Market.GovernmentBond)


def _uk() -> Calendar:
    """The USDLibor fixing calendar — UnitedKingdom(Exchange), via the index."""
    return USDLibor(_Q).fixing_calendar()


def _joint() -> Calendar:
    """probe.cpp:246-248 — JointCalendar(BMA fixing cal, USDLibor fixing cal)."""
    return JointCalendar([BMAIndex().fixing_calendar(), USDLibor(_Q).fixing_calendar()])


# Evaluation dates — probe.cpp:254-268.
_BIZ_DAY = Date.from_ymd(17, Month.January, 2024)  # Wednesday, open everywhere
_SATURDAY = Date.from_ymd(20, Month.January, 2024)
_SUNDAY = Date.from_ymd(30, Month.June, 2024)
_US_HOLIDAY = Date.from_ymd(4, Month.July, 2024)  # US shut, UK open
_UK_HOLIDAY = Date.from_ymd(26, Month.August, 2024)  # UK shut, US open
_BOTH_CLOSED = Date.from_ymd(1, Month.January, 2025)
_TARGET_HOLIDAY_START = Date.from_ymd(29, Month.April, 2024)


@dataclass(frozen=True)
class _Case:
    """One row of the probe's ``cases`` vector (probe.cpp:270-317)."""

    eval_date: Date
    tenor: Period
    settlement_days: int
    calendar: Calendar
    bma_period: Period
    bma_convention: BusinessDayConvention
    bma_day_count: DayCounter
    libor_tenor: Period


def _c(
    eval_date: Date,
    tenor: Period,
    settlement_days: int = 2,
    calendar: Calendar | None = None,
    bma_period: Period | None = None,
    bma_convention: BusinessDayConvention = _FOLLOWING,
    bma_day_count: DayCounter | None = None,
    libor_tenor: Period | None = None,
) -> _Case:
    return _Case(
        eval_date,
        tenor,
        settlement_days,
        calendar if calendar is not None else _joint(),
        bma_period if bma_period is not None else _Q,
        bma_convention,
        bma_day_count if bma_day_count is not None else _AA_ISDA,
        libor_tenor if libor_tenor is not None else _Q,
    )


CASES: dict[str, _Case] = {
    # --- tenor sweep on a plain business day -----------------------------
    "bma_1y": _c(_BIZ_DAY, Period(1, _YEARS)),
    "bma_2y": _c(_BIZ_DAY, Period(2, _YEARS)),
    "bma_5y": _c(_BIZ_DAY, Period(5, _YEARS)),
    "bma_10y": _c(_BIZ_DAY, Period(10, _YEARS)),
    "bma_18m": _c(_BIZ_DAY, Period(18, _MONTHS)),
    "bma_9m": _c(_BIZ_DAY, Period(9, _MONTHS)),
    # --- evaluation dates that are not business days ----------------------
    "bma_5y_saturday": _c(_SATURDAY, Period(5, _YEARS)),
    "bma_5y_sunday": _c(_SUNDAY, Period(5, _YEARS)),
    "bma_5y_us_holiday": _c(_US_HOLIDAY, Period(5, _YEARS)),
    "bma_5y_uk_holiday": _c(_UK_HOLIDAY, Period(5, _YEARS)),
    "bma_5y_both_closed": _c(_BOTH_CLOSED, Period(5, _YEARS)),
    "bma_us_cal_5y_us_holiday": _c(_US_HOLIDAY, Period(5, _YEARS), calendar=_us()),
    "bma_us_cal_5y_uk_holiday": _c(_UK_HOLIDAY, Period(5, _YEARS), calendar=_us()),
    "bma_uk_cal_5y_us_holiday": _c(_US_HOLIDAY, Period(5, _YEARS), calendar=_uk()),
    "bma_uk_cal_5y_uk_holiday": _c(_UK_HOLIDAY, Period(5, _YEARS), calendar=_uk()),
    # --- a helper calendar that is neither index's ------------------------
    "bma_target_cal_5y": _c(_BIZ_DAY, Period(5, _YEARS), calendar=TARGET()),
    "bma_target_cal_5y_saturday": _c(_SATURDAY, Period(5, _YEARS), calendar=TARGET()),
    "bma_target_cal_5y_target_holiday_maturity": _c(
        _TARGET_HOLIDAY_START, Period(5, _YEARS), calendar=TARGET()
    ),
    "bma_joint_cal_5y_target_holiday_maturity": _c(
        _TARGET_HOLIDAY_START, Period(5, _YEARS)
    ),
    # --- settlement days ---------------------------------------------------
    "bma_5y_settle0": _c(_BIZ_DAY, Period(5, _YEARS), settlement_days=0),
    "bma_5y_settle1": _c(_BIZ_DAY, Period(5, _YEARS), settlement_days=1),
    "bma_5y_settle3": _c(_BIZ_DAY, Period(5, _YEARS), settlement_days=3),
    "bma_5y_settle5": _c(_BIZ_DAY, Period(5, _YEARS), settlement_days=5),
    # --- BMA leg knobs -----------------------------------------------------
    "bma_5y_semiannual": _c(_BIZ_DAY, Period(5, _YEARS), bma_period=Period(6, _MONTHS)),
    "bma_5y_annual": _c(_BIZ_DAY, Period(5, _YEARS), bma_period=Period(1, _YEARS)),
    "bma_5y_modfollowing": _c(
        _BIZ_DAY, Period(5, _YEARS), bma_convention=BusinessDayConvention.ModifiedFollowing
    ),
    "bma_5y_preceding": _c(
        _BIZ_DAY, Period(5, _YEARS), bma_convention=BusinessDayConvention.Preceding
    ),
    "bma_5y_a360_bma_dc": _c(_BIZ_DAY, Period(5, _YEARS), bma_day_count=Actual360()),
    # --- Libor leg tenor ---------------------------------------------------
    "bma_5y_libor1m": _c(_BIZ_DAY, Period(5, _YEARS), libor_tenor=Period(1, _MONTHS)),
    "bma_5y_libor6m": _c(_BIZ_DAY, Period(5, _YEARS), libor_tenor=Period(6, _MONTHS)),
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/ts/ratehelpers")


@pytest.fixture(autouse=True)
def _global_state() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Restore the two pieces of global state these cases move.

    Each case sets the evaluation date itself (the probe does the same, per
    case, at probe.cpp:206) and clears the fixing histories before seeding
    (probe.cpp:207); both are undone here so no other module inherits them.
    """
    settings = ObservableSettings()
    previous = settings.evaluation_date
    try:
        yield
    finally:
        settings.evaluation_date = previous
        IndexManager().clear_histories()


def _flat(reference_date: Date, rate: float) -> YieldTermStructureProtocol:
    """probe.cpp — FlatForward(eval, rate, Actual365Fixed(), Continuous, Annual)."""
    return cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            reference_date, rate, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
        ),
    )


def _seed_fixings(index: InterestRateIndex, eval_date: Date) -> None:
    """probe.cpp:155-161 — ``rate(d) = 0.02 + 1e-5 * (serial % 100)``.

    Only dates STRICTLY before the evaluation date are seeded, so a fixing
    that falls on the evaluation date is always forecast rather than read out
    of history (``InterestRateIndex::fixing``).
    """
    d = eval_date - _SEED_DAYS_BACK
    while d < eval_date:
        if index.is_valid_fixing_date(d):
            index.add_fixing(d, 0.02 + 1.0e-5 * float(d.serial_number() % 100), True)
        d = d + 1


def _build(case: _Case) -> BMASwapRateHelper:
    """Mirror ``emitBma`` (probe.cpp:205-221) exactly, including the seeding."""
    ObservableSettings().evaluation_date = case.eval_date
    IndexManager().clear_histories()

    bma_index = BMAIndex()
    libor_index = USDLibor(case.libor_tenor, _flat(case.eval_date, _LIBOR_CURVE_RATE))
    _seed_fixings(bma_index, case.eval_date)
    _seed_fixings(libor_index, case.eval_date)

    helper = BMASwapRateHelper(
        _QUOTE,
        case.tenor,
        case.settlement_days,
        case.calendar,
        case.bma_period,
        case.bma_convention,
        case.bma_day_count,
        bma_index,
        libor_index,
    )
    helper.set_term_structure(_flat(case.eval_date, _BOOTSTRAPPED_RATE))
    return helper


def _iso(d: Date) -> str:
    return f"{d.year()}-{int(d.month()):02d}-{d.day_of_month():02d}"


def _assert_date(actual: Date, ref: dict[str, Any], name: str) -> None:
    """Serial AND ISO — EXACT tier, so a right serial with a wrong calendar fails."""
    assert actual.serial_number() == ref[f"{name}_serial"], f"{name}: serial"
    assert _iso(actual) == ref[f"{name}_iso"], f"{name}: ISO"


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(CASES))
def test_helper_dates(key: str, cpp: dict[str, Any]) -> None:
    """The five ``RateHelper`` date accessors. EXACT tier.

    C++ ``BMASwapRateHelper::initializeDates`` assigns only ``earliestDate_``
    and ``latestDate_``; ``maturityDate()``, ``latestRelevantDate()`` and
    ``pillarDate()`` therefore all fall through to ``latestDate()``
    (bootstraphelper.hpp:179-203). That fall-through is asserted here too — a
    port that "helpfully" sets ``maturityDate_`` to the swap maturity would
    move every curve node built with a maturity pillar.
    """
    ref = cpp[key]
    helper = _build(CASES[key])
    _assert_date(helper.earliest_date(), ref, "earliest")
    _assert_date(helper.latest_date(), ref, "latest")
    _assert_date(helper.maturity_date(), ref, "maturity")
    _assert_date(helper.latest_relevant_date(), ref, "latest_relevant")
    _assert_date(helper.pillar_date(), ref, "pillar")


@pytest.mark.parametrize("key", list(CASES))
def test_underlying_swap_schedules(key: str, cpp: dict[str, Any]) -> None:
    """The swap the dates were read off: both legs, first and last flow. EXACT tier.

    The two legs are rolled on the two INDEXES' calendars and conventions
    (ratehelpers.cpp:689-702), not on the helper's, so their sizes and boundary
    dates localise a schedule bug that ``latest_date`` alone would only show as
    an off-by-a-few-days.
    """
    ref = cpp[key]
    case = CASES[key]
    helper = _build(case)
    swap = helper.swap()

    _assert_date(swap.start_date(), ref, "swap_start")
    _assert_date(swap.maturity_date(), ref, "swap_maturity")

    # ratehelpers.cpp:716-717 — the adjusted maturity and its weekday select
    # the branch of the next-Wednesday roll.
    adjusted = case.calendar.adjust(swap.maturity_date(), _FOLLOWING)
    _assert_date(adjusted, ref, "adjusted_swap_maturity")
    assert int(adjusted.weekday()) == ref["adjusted_swap_maturity_weekday"]

    for name, leg in (("libor_leg", swap.libor_leg()), ("bma_leg", swap.bma_leg())):
        assert len(leg) == ref[f"{name}_size"], f"{name}: flow count"
        first, last = leg[0], leg[-1]
        # Both BMA-swap legs are pure coupon legs — no redemption flow — so the
        # accrual accessors are always available (bmaswap.cpp:43-59).
        assert isinstance(first, Coupon), f"{name}[0] is not a Coupon"
        assert isinstance(last, Coupon), f"{name}[-1] is not a Coupon"
        _assert_date(first.accrual_start_date(), ref, f"{name}_first_accrual_start")
        _assert_date(first.date(), ref, f"{name}_first_payment")
        _assert_date(last.accrual_end_date(), ref, f"{name}_last_accrual_end")
        _assert_date(last.date(), ref, f"{name}_last_payment")


def test_next_wednesday_roll_depends_on_helper_calendar(cpp: dict[str, Any]) -> None:
    """``latest_date`` moves a full week when the helper calendar shuts the maturity.

    Both cases start from 29 April 2024 with the same tenor and quote. With
    the joint US/UK calendar the swap matures on Tuesday 1 May 2029, which
    takes the ``w < 4`` branch and lands ``latest_date`` on 3 May. With TARGET
    the settlement advance already skips TARGET's Labour Day, so the maturity
    is Wednesday 2 May 2029 — the ``w >= 4`` branch, ``d + (11 - 4)``, and
    ``latest_date`` on 10 May. A port that returned the maturity, or that
    dropped the ``>= 4`` branch, cannot produce both.
    """
    target_ref = cpp["bma_target_cal_5y_target_holiday_maturity"]
    joint_ref = cpp["bma_joint_cal_5y_target_holiday_maturity"]
    assert target_ref["latest_serial"] - joint_ref["latest_serial"] == 7
    assert target_ref["adjusted_swap_maturity_weekday"] == 4
    assert joint_ref["adjusted_swap_maturity_weekday"] == 3

    target = _build(CASES["bma_target_cal_5y_target_holiday_maturity"])
    joint = _build(CASES["bma_joint_cal_5y_target_holiday_maturity"])
    assert target.latest_date() - joint.latest_date() == 7
    assert target.latest_date() > target.swap().maturity_date()


def test_wednesday_maturity_rolls_a_full_week(cpp: dict[str, Any]) -> None:
    """A maturity already on a Wednesday jumps seven days, not zero.

    ``bma_5y_saturday`` matures on Wednesday 24 January 2029; the BMA index
    value date of the *next* Wednesday is 1 February 2029, eight days later.
    """
    ref = cpp["bma_5y_saturday"]
    assert ref["adjusted_swap_maturity_weekday"] == 4
    assert ref["latest_serial"] - ref["swap_maturity_serial"] == 8
    helper = _build(CASES["bma_5y_saturday"])
    assert helper.latest_date() - helper.swap().maturity_date() == 8


def test_dates_re_initialize_when_the_evaluation_date_moves(cpp: dict[str, Any]) -> None:
    """This is a ``RelativeDateRateHelper``: moving the eval date re-derives the dates.

    ``bma_5y`` and ``bma_5y_saturday`` differ ONLY in the evaluation date, so a
    helper built for the first and then re-pointed at the second's date has to
    reproduce the second's reference dates exactly — the same values a freshly
    built helper gets. A port that computed the dates once in ``__init__`` and
    never registered with ``Settings`` would keep the January-17 answers.
    """
    helper = _build(CASES["bma_5y"])
    ref_weekday = cpp["bma_5y"]
    _assert_date(helper.earliest_date(), ref_weekday, "earliest")

    ObservableSettings().evaluation_date = _SATURDAY
    ref_saturday = cpp["bma_5y_saturday"]
    _assert_date(helper.earliest_date(), ref_saturday, "earliest")
    _assert_date(helper.latest_date(), ref_saturday, "latest")
    _assert_date(helper.pillar_date(), ref_saturday, "pillar")


def test_joint_calendar_roll_is_not_the_helper_calendar(cpp: dict[str, Any]) -> None:
    """The reference-date roll uses the JOINT calendar, the advance does not.

    On 4 July 2024 the US is closed and the UK is open. With the helper
    calendar set to the UK one the joint calendar is UK-only, so the reference
    date stays on the 4th and settlement lands on the 8th; with the US in the
    joint calendar it rolls to the 5th and settlement lands on the 9th. A port
    that used one calendar for both steps produces the same answer twice.
    """
    uk_only = cpp["bma_uk_cal_5y_us_holiday"]
    with_us = cpp["bma_5y_us_holiday"]
    assert with_us["earliest_serial"] - uk_only["earliest_serial"] == 1

    assert (
        _build(CASES["bma_5y_us_holiday"]).earliest_date()
        - _build(CASES["bma_uk_cal_5y_us_holiday"]).earliest_date()
        == 1
    )


# ---------------------------------------------------------------------------
# Implied quote
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(CASES))
def test_implied_quote(key: str, cpp: dict[str, Any]) -> None:
    """``swap.fair_libor_fraction()`` and the two leg NPVs behind it. TIGHT tier.

    The BMA leg forecasts off the curve passed to ``set_term_structure`` while
    the Libor leg and the discounting use the ibor index's own curve
    (ratehelpers.cpp:713-714); the two flat rates differ (2% vs 3.5%) so a port
    that wired either leg to the wrong curve fails here rather than agreeing by
    coincidence.
    """
    ref = cpp[key]
    helper = _build(CASES[key])
    quote = helper.implied_quote()
    tight(quote, ref["implied_quote"], reason=f"{key} implied quote")
    tight(helper.quote_error(), ref["quote_error"], reason=f"{key} quote error")

    swap = helper.swap()
    tight(swap.npv(), ref["swap_npv"], reason=f"{key} swap NPV")
    tight(swap.libor_leg_npv(), ref["swap_libor_leg_npv"], reason=f"{key} Libor-leg NPV")
    tight(swap.bma_leg_npv(), ref["swap_bma_leg_npv"], reason=f"{key} BMA-leg NPV")


def test_quote_error_is_quote_minus_implied(cpp: dict[str, Any]) -> None:
    """``quote_error() == quote - implied_quote()``, at the quoted 0.70."""
    helper = _build(CASES["bma_5y"])
    exact(helper.quote().value(), _QUOTE)
    tight(helper.quote_error(), _QUOTE - helper.implied_quote())
    tight(cpp["bma_5y"]["quote_error"], _QUOTE - cpp["bma_5y"]["implied_quote"])


def test_implied_quote_requires_a_term_structure() -> None:
    """C++ ``QL_REQUIRE(termStructure_ != nullptr, ...)`` (ratehelpers.cpp:737)."""
    case = CASES["bma_5y"]
    ObservableSettings().evaluation_date = case.eval_date
    IndexManager().clear_histories()
    helper = BMASwapRateHelper(
        _QUOTE,
        case.tenor,
        case.settlement_days,
        case.calendar,
        case.bma_period,
        case.bma_convention,
        case.bma_day_count,
        BMAIndex(),
        USDLibor(case.libor_tenor, _flat(case.eval_date, _LIBOR_CURVE_RATE)),
    )
    with pytest.raises(LibraryException, match="term structure not set"):
        helper.implied_quote()
