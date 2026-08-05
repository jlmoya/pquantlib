"""Tests for the v1.43 multiple-resets coupon family.

Probe source: migration-harness/cpp/probes/v143_cf_multipleresets/probe.cpp
Reference:    migration-harness/references/v143/cf/multipleresets.json

Everything is rebuilt here from the same literal tables the probe uses — an
explicit zero curve, an explicit IborIndex and explicit date-list Schedules —
so a failure localises to the coupon/leg code rather than to a shared fixture.
``test_fixture_matches_cpp`` asserts the tables themselves first.

Three legs are cross-validated coupon-by-coupon (not just by NPV, which can
match while two errors cancel): a Compound leg using every vector-form setter
plus a NON-ZERO payment lag, a Simple leg using every scalar-form setter, and a
one-coupon leg whose ex-coupon date exercises the end-of-month roll.

``test_setter_has_an_observable_effect`` is the anti-regression guard for the
defect this subsystem is prone to (an optional argument that is accepted,
stored and then never passed on — see the payment lags dropped by
FixedVsFloatingSwap / OvernightIndexedSwap): it flips exactly one setter away
from its default and asserts the observable changes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.cashflows.ibor_coupon import IborCoupon
from pquantlib.cashflows.multiple_resets_coupon import (
    AveragingMultipleResetsPricer,
    CompoundingMultipleResetsPricer,
    MultipleResetsCoupon,
    MultipleResetsLeg,
    MultipleResetsPricer,
)
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.yield_.interpolated_zero_curve import InterpolatedZeroCurve
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.calendars.united_states import UnitedStates
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

# --- the literal fixture tables (identical to the probe's) -----------------

_D = Date.from_ymd
_TODAY = _D(5, Month.January, 2026)

_CURVE_DATES = [
    _D(5, Month.January, 2026),
    _D(5, Month.July, 2026),
    _D(5, Month.January, 2027),
    _D(5, Month.January, 2028),
    _D(5, Month.January, 2030),
]
_CURVE_ZEROS = [0.0200, 0.0245, 0.0280, 0.0310, 0.0335]

_SCHED_A = [
    _D(20, Month.January, 2026),
    _D(20, Month.February, 2026),
    _D(20, Month.March, 2026),
    _D(20, Month.April, 2026),
    _D(20, Month.May, 2026),
    _D(22, Month.June, 2026),
    _D(20, Month.July, 2026),
    _D(20, Month.August, 2026),
    _D(21, Month.September, 2026),
    _D(20, Month.October, 2026),
    _D(20, Month.November, 2026),
    _D(21, Month.December, 2026),
    _D(20, Month.January, 2027),
]
_SCHED_B = [
    _D(19, Month.February, 2026),
    _D(20, Month.April, 2026),
    _D(19, Month.June, 2026),
    _D(20, Month.August, 2026),
    _D(11, Month.November, 2026),
]
_SCHED_EOM = [
    _D(30, Month.January, 2026),
    _D(27, Month.February, 2026),
    _D(31, Month.March, 2026),
]

_US = UnitedStates(UnitedStates.Market.GovernmentBond)


def _schedule(dates: Sequence[Date]) -> Schedule:
    return Schedule(dates, calendar=TARGET(), convention=BusinessDayConvention.ModifiedFollowing)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/cf/multipleresets")


@pytest.fixture(scope="module")
def curve() -> InterpolatedZeroCurve:
    return InterpolatedZeroCurve(_CURVE_DATES, _CURVE_ZEROS, Actual365Fixed())


@pytest.fixture(scope="module")
def index(curve: InterpolatedZeroCurve) -> IborIndex:
    return IborIndex(
        "MRTestIbor",
        Period(1, TimeUnit.Months),
        2,
        EURCurrency(),
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        False,
        Actual360(),
        curve,
    )


# --- leg builders (mirroring the probe's) ----------------------------------


def _leg_compound(
    index: IborIndex,
    *,
    ex_period: Period | None = None,
    ex_calendar: Calendar | None = None,
    ex_convention: BusinessDayConvention = BusinessDayConvention.Unadjusted,
    ex_end_of_month: bool = False,
) -> list[CashFlow]:
    """Every vector-form setter, a non-zero payment lag, Compound averaging."""
    builder = MultipleResetsLeg(_schedule(_SCHED_A), index, 3)
    builder.with_notionals([1_000_000.0, 2_000_000.0, 3_000_000.0, 4_000_000.0])
    builder.with_payment_day_counter(Thirty360(Thirty360Convention.BondBasis))
    builder.with_payment_adjustment(BusinessDayConvention.Preceding)
    builder.with_payment_calendar(_US)
    builder.with_payment_lag(3)
    builder.with_fixing_days([0, 1, 2, 5])
    builder.with_gearings([1.0, 1.5, 0.8, 2.0])
    builder.with_coupon_spreads([0.0010, 0.0020, -0.0005, 0.0])
    builder.with_rate_spreads([0.0001, 0.0002, 0.0003, 0.0004])
    builder.with_averaging_method(RateAveraging.Compound)
    if ex_period is not None:
        builder.with_ex_coupon_period(ex_period, ex_calendar, ex_convention, ex_end_of_month)
    return builder.build()


def _leg_simple(index: IborIndex) -> list[CashFlow]:
    """Every scalar-form setter, Simple averaging, empty ex-coupon calendar.

    The payment lag stays 0 on purpose: ``Calendar.advance`` ignores the
    business-day convention when the lag is non-zero and the unit is Days, so
    ``with_payment_adjustment`` is only observable at lag 0. Both coupon end
    dates are US-GovernmentBond holidays but TARGET business days, so the
    payment calendar and the adjustment each move the payment date.
    """
    builder = MultipleResetsLeg(_schedule(_SCHED_B), index, 2)
    builder.with_notionals(5_000_000.0)
    builder.with_payment_adjustment(BusinessDayConvention.Preceding)
    builder.with_payment_calendar(_US)
    builder.with_fixing_days(1)
    builder.with_gearings(0.75)
    builder.with_coupon_spreads(0.0025)
    builder.with_rate_spreads(0.0015)
    builder.with_ex_coupon_period(Period(1, TimeUnit.Weeks), None, BusinessDayConvention.Preceding, False)
    builder.with_averaging_method(RateAveraging.Simple)
    return builder.build()


def _leg_eom(index: IborIndex, *, end_of_month: bool) -> list[CashFlow]:
    """One coupon whose payment date is a month end (the end-of-month roll)."""
    builder = MultipleResetsLeg(_schedule(_SCHED_EOM), index, 2)
    builder.with_notionals(1_000_000.0)
    builder.with_payment_adjustment(BusinessDayConvention.Unadjusted)
    builder.with_ex_coupon_period(
        Period(1, TimeUnit.Months),
        TARGET(),
        BusinessDayConvention.Following,
        end_of_month,
    )
    return builder.build()


# --- helpers ---------------------------------------------------------------


def _coupons(leg: Sequence[CashFlow]) -> list[MultipleResetsCoupon]:
    out: list[MultipleResetsCoupon] = []
    for cf in leg:
        assert isinstance(cf, MultipleResetsCoupon)
        out.append(cf)
    return out


def _serials(dates: Sequence[Date]) -> list[int]:
    return [d.serial_number() for d in dates]


def _check_leg(
    leg: Sequence[CashFlow],
    ref: dict[str, Any],
    curve: InterpolatedZeroCurve,
    index: IborIndex,
) -> None:
    assert len(leg) == ref["size"]
    tolerance.tight(CashFlows.npv_curve(leg, curve, False, _TODAY, _TODAY), ref["npv"])

    for coupon, exp in zip(_coupons(leg), ref["coupons"], strict=True):
        assert coupon.date().serial_number() == exp["payment_date_serial"]
        tolerance.tight(coupon.nominal(), exp["nominal"])
        assert coupon.accrual_start_date().serial_number() == exp["accrual_start_serial"]
        assert coupon.accrual_end_date().serial_number() == exp["accrual_end_serial"]
        assert coupon.reference_period_start().serial_number() == exp["ref_period_start_serial"]
        assert coupon.reference_period_end().serial_number() == exp["ref_period_end_serial"]
        assert coupon.ex_coupon_date().serial_number() == exp["ex_coupon_date_serial"]
        assert coupon.day_counter().name() == exp["day_counter"]
        tolerance.tight(coupon.accrual_period(), exp["accrual_period"])
        assert coupon.accrual_days() == exp["accrual_days"]
        assert coupon.fixing_days() == exp["fixing_days"]
        tolerance.tight(coupon.gearing(), exp["gearing"])
        tolerance.tight(coupon.spread(), exp["spread"])
        tolerance.tight(coupon.rate_spread(), exp["rate_spread"])
        assert coupon.fixing_date().serial_number() == exp["fixing_date_serial"]
        assert _serials(coupon.value_dates()) == exp["value_date_serials"]
        assert _serials(coupon.fixing_dates()) == exp["fixing_date_serials"]
        for dt, exp_dt in zip(coupon.dt(), exp["dt"], strict=True):
            tolerance.tight(dt, exp_dt)
        fixings = [index.fixing(d, False) for d in coupon.fixing_dates()]
        for fixing, exp_fixing in zip(fixings, exp["sub_period_fixings"], strict=True):
            tolerance.tight(fixing, exp_fixing)
        tolerance.tight(coupon.rate(), exp["rate"])
        tolerance.tight(coupon.amount(), exp["amount"])


# --- fixture echo ----------------------------------------------------------


def test_fixture_matches_cpp(cpp: dict[str, Any], index: IborIndex) -> None:
    setup = cpp["setup"]
    assert _TODAY.serial_number() == setup["evaluation_date_serial"]
    assert _serials(_CURVE_DATES) == setup["curve_date_serials"]
    assert setup["curve_zeros"] == _CURVE_ZEROS
    assert index.name() == setup["index_name"]
    assert index.fixing_days() == setup["index_fixing_days"]
    assert index.day_counter().name() == setup["index_day_counter"]
    assert _serials(_SCHED_A) == setup["schedule_a_serials"]
    assert _serials(_SCHED_B) == setup["schedule_b_serials"]
    assert _serials(_SCHED_EOM) == setup["schedule_eom_serials"]


# --- the three cross-validated legs ----------------------------------------


def test_leg_compound(cpp: dict[str, Any], curve: InterpolatedZeroCurve, index: IborIndex) -> None:
    leg = _leg_compound(
        index,
        ex_period=Period(6, TimeUnit.Days),
        ex_calendar=_US,
        ex_convention=BusinessDayConvention.Preceding,
    )
    _check_leg(leg, cpp["leg_compound"], curve, index)


def test_leg_simple(cpp: dict[str, Any], curve: InterpolatedZeroCurve, index: IborIndex) -> None:
    _check_leg(_leg_simple(index), cpp["leg_simple"], curve, index)


def test_leg_eom(cpp: dict[str, Any], curve: InterpolatedZeroCurve, index: IborIndex) -> None:
    _check_leg(_leg_eom(index, end_of_month=True), cpp["leg_eom"], curve, index)


def test_call_delegates_to_build(index: IborIndex) -> None:
    builder = MultipleResetsLeg(_schedule(_SCHED_EOM), index, 2)
    builder.with_notionals(1_000_000.0)
    assert _serials([cf.date() for cf in builder()]) == _serials([cf.date() for cf in builder.build()])


# --- ex-coupon: calendar / convention / period / end-of-month all bite ------


def _ex_coupon_legs(index: IborIndex) -> dict[str, list[CashFlow]]:
    week = Period(1, TimeUnit.Weeks)
    six_days = Period(6, TimeUnit.Days)
    preceding = BusinessDayConvention.Preceding
    following = BusinessDayConvention.Following
    return {
        "a_none": _leg_compound(index),
        "a_6d_us_preceding": _leg_compound(
            index, ex_period=six_days, ex_calendar=_US, ex_convention=preceding
        ),
        "a_6d_empty_preceding": _leg_compound(
            index, ex_period=six_days, ex_calendar=None, ex_convention=preceding
        ),
        "a_1w_target_preceding": _leg_compound(
            index, ex_period=week, ex_calendar=TARGET(), ex_convention=preceding
        ),
        "a_1w_us_preceding": _leg_compound(index, ex_period=week, ex_calendar=_US, ex_convention=preceding),
        "a_1w_us_following": _leg_compound(index, ex_period=week, ex_calendar=_US, ex_convention=following),
        "eom_1m_true": _leg_eom(index, end_of_month=True),
        "eom_1m_false": _leg_eom(index, end_of_month=False),
    }


@pytest.mark.parametrize(
    "key",
    [
        "a_none",
        "a_6d_us_preceding",
        "a_6d_empty_preceding",
        "a_1w_target_preceding",
        "a_1w_us_preceding",
        "a_1w_us_following",
        "eom_1m_true",
        "eom_1m_false",
    ],
)
def test_ex_coupon_matrix(cpp: dict[str, Any], index: IborIndex, key: str) -> None:
    leg = _ex_coupon_legs(index)[key]
    assert _serials([c.ex_coupon_date() for c in _coupons(leg)]) == cpp["ex_coupon_matrix"][key]


def test_ex_coupon_matrix_rows_are_discriminating(cpp: dict[str, Any], index: IborIndex) -> None:
    """The matrix is only a guard if its rows actually differ.

    Each pair below isolates one ex-coupon argument, so a port that accepted it
    and dropped it could not pass every row.
    """
    matrix = cpp["ex_coupon_matrix"]
    assert matrix["a_6d_us_preceding"] != matrix["a_6d_empty_preceding"]  # calendar
    assert matrix["a_1w_us_preceding"] != matrix["a_1w_target_preceding"]  # calendar
    assert matrix["a_1w_us_preceding"] != matrix["a_1w_us_following"]  # convention
    assert matrix["a_1w_us_preceding"] != matrix["a_6d_us_preceding"]  # period
    assert matrix["eom_1m_true"] != matrix["eom_1m_false"]  # end-of-month
    assert all(s == 0 for s in matrix["a_none"])  # no period -> null date
    legs = _ex_coupon_legs(index)
    assert all(c.ex_coupon_date() == Date() for c in _coupons(legs["a_none"]))


# --- every setter must have an observable effect ---------------------------


def _base_builder(index: IborIndex) -> MultipleResetsLeg:
    """Two-coupon all-defaults leg on schedule B (ends on US holidays)."""
    return MultipleResetsLeg(_schedule(_SCHED_B), index, 2).with_notionals(1_000_000.0)


def _payment_serials(leg: Sequence[CashFlow]) -> list[int]:
    return _serials([cf.date() for cf in leg])


def _rates(leg: Sequence[CashFlow]) -> list[float]:
    return [c.rate() for c in _coupons(leg)]


def _nominals(leg: Sequence[CashFlow]) -> list[float]:
    return [c.nominal() for c in _coupons(leg)]


def _accrual_periods(leg: Sequence[CashFlow]) -> list[float]:
    return [c.accrual_period() for c in _coupons(leg)]


def _fixing_serials(leg: Sequence[CashFlow]) -> list[list[int]]:
    return [_serials(c.fixing_dates()) for c in _coupons(leg)]


def _ex_coupon_serials(leg: Sequence[CashFlow]) -> list[int]:
    return _serials([c.ex_coupon_date() for c in _coupons(leg)])


_SETTER_CASES: list[
    tuple[
        str,
        Callable[[MultipleResetsLeg], MultipleResetsLeg],
        Callable[[Sequence[CashFlow]], object],
    ]
] = [
    (
        "with_notionals",
        lambda b: b.with_notionals([1_000_000.0, 7_000_000.0]),
        _nominals,
    ),
    (
        "with_payment_day_counter",
        lambda b: b.with_payment_day_counter(Thirty360(Thirty360Convention.BondBasis)),
        _accrual_periods,
    ),
    (
        "with_payment_adjustment",
        lambda b: b.with_payment_calendar(_US).with_payment_adjustment(BusinessDayConvention.Preceding),
        _payment_serials,
    ),
    ("with_payment_calendar", lambda b: b.with_payment_calendar(_US), _payment_serials),
    ("with_payment_lag", lambda b: b.with_payment_lag(2), _payment_serials),
    ("with_fixing_days", lambda b: b.with_fixing_days(5), _fixing_serials),
    ("with_fixing_days_vector", lambda b: b.with_fixing_days([5, 7]), _fixing_serials),
    ("with_gearings", lambda b: b.with_gearings(2.0), _rates),
    ("with_gearings_vector", lambda b: b.with_gearings([2.0, 3.0]), _rates),
    ("with_coupon_spreads", lambda b: b.with_coupon_spreads(0.01), _rates),
    ("with_coupon_spreads_vector", lambda b: b.with_coupon_spreads([0.01, 0.02]), _rates),
    ("with_rate_spreads", lambda b: b.with_rate_spreads(0.01), _rates),
    ("with_rate_spreads_vector", lambda b: b.with_rate_spreads([0.01, 0.02]), _rates),
    (
        "with_ex_coupon_period",
        lambda b: b.with_ex_coupon_period(
            Period(1, TimeUnit.Weeks), TARGET(), BusinessDayConvention.Preceding, False
        ),
        _ex_coupon_serials,
    ),
    (
        "with_averaging_method",
        lambda b: b.with_averaging_method(RateAveraging.Simple),
        _rates,
    ),
]


@pytest.mark.parametrize(
    ("name", "mutate", "observe"),
    _SETTER_CASES,
    ids=[case[0] for case in _SETTER_CASES],
)
def test_setter_has_an_observable_effect(
    index: IborIndex,
    name: str,
    mutate: Callable[[MultipleResetsLeg], MultipleResetsLeg],
    observe: Callable[[Sequence[CashFlow]], object],
) -> None:
    """Flipping one setter off its default must change what the leg produces.

    This is the direct guard against an argument that is accepted, stored and
    then silently never used.
    """
    baseline = observe(_base_builder(index).build())
    variant = observe(mutate(_base_builder(index)).build())
    assert variant != baseline, f"{name} had no observable effect"


def test_setters_return_self_for_chaining(index: IborIndex) -> None:
    builder = MultipleResetsLeg(_schedule(_SCHED_B), index, 2)
    assert builder.with_notionals(1.0) is builder
    assert builder.with_payment_day_counter(Actual360()) is builder
    assert builder.with_payment_adjustment(BusinessDayConvention.Following) is builder
    assert builder.with_payment_calendar(TARGET()) is builder
    assert builder.with_payment_lag(0) is builder
    assert builder.with_fixing_days(2) is builder
    assert builder.with_gearings(1.0) is builder
    assert builder.with_coupon_spreads(0.0) is builder
    assert builder.with_rate_spreads(0.0) is builder
    assert builder.with_ex_coupon_period(Period(), None, BusinessDayConvention.Unadjusted, False) is builder
    assert builder.with_averaging_method(RateAveraging.Compound) is builder


def test_scalar_setter_applies_to_every_coupon(index: IborIndex) -> None:
    """A scalar setter is stored as a one-element vector and reused (detail::get)."""
    leg = _base_builder(index).with_gearings(1.25).build()
    assert [c.gearing() for c in _coupons(leg)] == [1.25, 1.25]


def test_vector_setter_shorter_than_the_leg_reuses_the_last_entry(
    index: IborIndex,
) -> None:
    """``detail::get`` falls back to ``back()``, not to the default."""
    builder = MultipleResetsLeg(_schedule(_SCHED_A), index, 3)  # 4 coupons
    leg = builder.with_notionals([1_000_000.0, 2_000_000.0]).build()
    assert _nominals(leg) == [1_000_000.0, 2_000_000.0, 2_000_000.0, 2_000_000.0]


def test_fixing_days_default_to_the_index(index: IborIndex) -> None:
    leg = _base_builder(index).build()
    assert [c.fixing_days() for c in _coupons(leg)] == [index.fixing_days()] * 2


def test_payment_day_counter_defaults_to_the_index(index: IborIndex) -> None:
    leg = _base_builder(index).build()
    assert all(c.day_counter().name() == index.day_counter().name() for c in _coupons(leg))


# --- pricers ---------------------------------------------------------------


def test_base_pricer_is_abstract() -> None:
    """C++ leaves ``swapletRate`` pure virtual on MultipleResetsPricer."""
    with pytest.raises(TypeError):
        MultipleResetsPricer()  # type: ignore[abstract]


_UNIMPLEMENTED_CASES: list[tuple[str, Callable[[MultipleResetsPricer], float]]] = [
    ("swaplet_price", lambda p: p.swaplet_price()),
    ("caplet_price", lambda p: p.caplet_price(0.03)),
    ("caplet_rate", lambda p: p.caplet_rate(0.03)),
    ("floorlet_price", lambda p: p.floorlet_price(0.01)),
    ("floorlet_rate", lambda p: p.floorlet_rate(0.01)),
]


@pytest.mark.parametrize(
    ("key", "call"),
    _UNIMPLEMENTED_CASES,
    ids=[case[0] for case in _UNIMPLEMENTED_CASES],
)
def test_pricer_unimplemented_methods_raise(
    cpp: dict[str, Any],
    index: IborIndex,
    key: str,
    call: Callable[[MultipleResetsPricer], float],
) -> None:
    assert cpp["pricer_failures"][key]["raises"] is True
    coupon = _coupons(
        _leg_compound(
            index,
            ex_period=Period(6, TimeUnit.Days),
            ex_calendar=_US,
            ex_convention=BusinessDayConvention.Preceding,
        )
    )[0]
    pricer = CompoundingMultipleResetsPricer()
    pricer.initialize(coupon)
    with pytest.raises(LibraryException):
        call(pricer)


def test_pricer_initialize_rejects_a_foreign_coupon(cpp: dict[str, Any], index: IborIndex) -> None:
    assert cpp["pricer_failures"]["initialize_wrong_coupon_type"]["raises"] is True
    other = IborCoupon(
        _D(20, Month.April, 2026),
        1_000_000.0,
        _D(20, Month.January, 2026),
        _D(20, Month.April, 2026),
        2,
        index,
    )
    with pytest.raises(LibraryException):
        AveragingMultipleResetsPricer().initialize(other)


def test_pricer_before_initialize_raises() -> None:
    with pytest.raises(LibraryException):
        CompoundingMultipleResetsPricer().swaplet_rate()


# --- constructor / build guards --------------------------------------------


def _guard_cases(index: IborIndex) -> dict[str, Callable[[], object]]:
    sched_a = _schedule(_SCHED_A)
    return {
        "ctor_no_index": lambda: MultipleResetsLeg(sched_a, None, 3),
        "ctor_empty_schedule": lambda: MultipleResetsLeg(Schedule([]), index, 3),
        "ctor_resets_do_not_divide": lambda: MultipleResetsLeg(sched_a, index, 5),
        "build_no_notional": lambda: MultipleResetsLeg(sched_a, index, 3).build(),
        "build_too_many_notionals": lambda: (
            MultipleResetsLeg(sched_a, index, 3).with_notionals([1.0, 2.0, 3.0, 4.0, 5.0]).build()
        ),
        "build_too_many_gearings": lambda: (
            MultipleResetsLeg(sched_a, index, 3)
            .with_notionals(1_000_000.0)
            .with_gearings([1.0, 1.0, 1.0, 1.0, 1.0])
            .build()
        ),
        "build_too_many_coupon_spreads": lambda: (
            MultipleResetsLeg(sched_a, index, 3)
            .with_notionals(1_000_000.0)
            .with_coupon_spreads([0.0, 0.0, 0.0, 0.0, 0.0])
            .build()
        ),
        "build_too_many_rate_spreads": lambda: (
            MultipleResetsLeg(sched_a, index, 3)
            .with_notionals(1_000_000.0)
            .with_rate_spreads([0.0, 0.0, 0.0, 0.0, 0.0])
            .build()
        ),
        "build_too_many_fixing_days": lambda: (
            MultipleResetsLeg(sched_a, index, 3)
            .with_notionals(1_000_000.0)
            .with_fixing_days([2, 2, 2, 2, 2])
            .build()
        ),
    }


@pytest.mark.parametrize(
    "key",
    [
        "ctor_no_index",
        "ctor_empty_schedule",
        "ctor_resets_do_not_divide",
        "build_no_notional",
        "build_too_many_notionals",
        "build_too_many_gearings",
        "build_too_many_coupon_spreads",
        "build_too_many_rate_spreads",
        "build_too_many_fixing_days",
    ],
)
def test_guards_raise(cpp: dict[str, Any], index: IborIndex, key: str) -> None:
    assert cpp["raises"][key]["raises"] is True
    with pytest.raises(LibraryException):
        _guard_cases(index)[key]()


def test_exactly_n_of_each_vector_is_accepted(index: IborIndex) -> None:
    """The guards are ``<= n``, so n entries must still build."""
    leg = (
        MultipleResetsLeg(_schedule(_SCHED_A), index, 3)
        .with_notionals([1.0, 2.0, 3.0, 4.0])
        .with_gearings([1.0, 1.0, 1.0, 1.0])
        .with_coupon_spreads([0.0, 0.0, 0.0, 0.0])
        .with_rate_spreads([0.0, 0.0, 0.0, 0.0])
        .with_fixing_days([2, 2, 2, 2])
        .build()
    )
    assert len(leg) == 4
