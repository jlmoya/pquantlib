"""Tests for FloatingRateCoupon + IborCoupon + OvernightIndexedCoupon + legs +
pricers (IborCouponPricer / BlackIborCouponPricer / CompoundingOvernightIndexedCouponPricer).
"""

from __future__ import annotations

import pytest

from pquantlib.cashflows.coupon_pricer import (
    BlackIborCouponPricer,
    CouponPricer,
    IborCouponPricer,
    set_coupon_pricer,
)
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.cashflows.ibor_coupon import IborCoupon
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.cashflows.overnight_indexed_coupon import (
    CompoundingOvernightIndexedCouponPricer,
    OvernightIndexedCoupon,
)
from pquantlib.cashflows.overnight_leg import overnight_leg
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.calendars.weekends_only import WeekendsOnly
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

# --- mock concrete indexes -------------------------------------------------


class _FlatIborIndex:
    """Mock IborIndexProtocol that always returns the same fixing."""

    def __init__(self, fixing: float, fixing_days: int = 2) -> None:
        self._fixing = fixing
        self._fixing_days = fixing_days

    def name(self) -> str:
        return "MOCK3M"

    def tenor(self) -> Period:
        return Period(3, TimeUnit.Months)

    def fixing_days(self) -> int:
        return self._fixing_days

    def currency(self) -> object:
        return None

    def fixing_calendar(self) -> Calendar:
        return NullCalendar()

    def day_counter(self) -> DayCounter:
        return Actual360()

    def business_day_convention(self) -> BusinessDayConvention:
        return BusinessDayConvention.ModifiedFollowing

    def end_of_month(self) -> bool:
        return False

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        del fixing_date, forecast_todays_fixing
        return self._fixing


class _FlatOvernightIndex:
    """Mock OvernightIndexProtocol returning a constant daily fixing."""

    def __init__(self, fixing: float) -> None:
        self._fixing = fixing

    def name(self) -> str:
        return "MOCKOIS"

    def currency(self) -> object:
        return None

    def fixing_calendar(self) -> Calendar:
        return WeekendsOnly()

    def day_counter(self) -> DayCounter:
        return Actual360()

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        del fixing_date, forecast_todays_fixing
        return self._fixing


# --- reference fixtures ----------------------------------------------------


@pytest.fixture(scope="module")
def ref_ibor() -> dict[str, float]:
    return reference_reader.load("cluster/l2d")["ibor_coupon"]


@pytest.fixture(scope="module")
def ref_oic() -> dict[str, float]:
    return reference_reader.load("cluster/l2d")["overnight_coupon"]


# --- IborCoupon -----------------------------------------------------------


def test_ibor_coupon_constant_fixing(ref_ibor: dict[str, float]) -> None:
    """At a flat 3.5% fixing, amount = 100K * 0.035 * 0.25 (90/360) = 875.

    Note: the C++ probe's reference value (872.286...) uses a
    real FlatForward(Continuous/Annual) curve where the forecast
    fixing diverges slightly from the curve's input rate due to the
    Continuous compounding. Our mock returns a flat 3.5% directly,
    so the IborCoupon test here verifies the rate / amount algebra
    against an analytical value rather than the curve-derived probe
    value. Probe's accrual_period (0.25) is verified instead.
    """
    d1 = Date.from_ymd(1, Month.January, 2026)
    d2 = Date.from_ymd(1, Month.April, 2026)
    idx = _FlatIborIndex(0.035, fixing_days=0)
    ic = IborCoupon(d2, 100_000.0, d1, d2, 0, idx, 1.0, 0.0)
    ic.set_pricer(BlackIborCouponPricer())
    # accrual_period = 90/360 = 0.25 (NullCalendar, no holiday adjustment)
    tolerance.tight(ic.accrual_period(), ref_ibor["accrual_period"])
    # rate = 0.035 (gearing 1, spread 0, no convexity)
    tolerance.tight(ic.rate(), 0.035)
    # amount = 100K * 0.035 * 0.25 = 875
    tolerance.tight(ic.amount(), 875.0)


def test_ibor_coupon_requires_pricer() -> None:
    d1 = Date.from_ymd(1, Month.January, 2026)
    d2 = Date.from_ymd(1, Month.April, 2026)
    idx = _FlatIborIndex(0.035)
    ic = IborCoupon(d2, 100_000.0, d1, d2, 2, idx)
    # rate() without pricer raises
    with pytest.raises(LibraryException, match="pricer not set"):
        ic.rate()


def test_ibor_coupon_index_fixing_passes_through() -> None:
    d1 = Date.from_ymd(1, Month.January, 2026)
    d2 = Date.from_ymd(1, Month.April, 2026)
    idx = _FlatIborIndex(0.0425, fixing_days=0)
    ic = IborCoupon(d2, 100_000.0, d1, d2, 0, idx)
    tolerance.exact(ic.index_fixing(), 0.0425)


def test_ibor_coupon_gearing_and_spread() -> None:
    """rate = gearing * fixing + spread."""
    d1 = Date.from_ymd(1, Month.January, 2026)
    d2 = Date.from_ymd(1, Month.April, 2026)
    idx = _FlatIborIndex(0.05, fixing_days=0)
    ic = IborCoupon(d2, 100_000.0, d1, d2, 0, idx, gearing=2.0, spread=0.01)
    ic.set_pricer(IborCouponPricer())
    # rate = 2.0 * 0.05 + 0.01 = 0.11
    tolerance.tight(ic.rate(), 0.11)


def test_floating_rate_coupon_rejects_null_gearing() -> None:
    d1 = Date.from_ymd(1, Month.January, 2026)
    d2 = Date.from_ymd(1, Month.April, 2026)
    idx = _FlatIborIndex(0.05)
    with pytest.raises(LibraryException, match="Null gearing"):
        IborCoupon(d2, 100_000.0, d1, d2, 2, idx, gearing=0.0)


def test_ibor_coupon_is_floating_rate_coupon() -> None:
    d1 = Date.from_ymd(1, Month.January, 2026)
    d2 = Date.from_ymd(1, Month.April, 2026)
    ic = IborCoupon(d2, 100_000.0, d1, d2, 2, _FlatIborIndex(0.035))
    assert isinstance(ic, FloatingRateCoupon)


# --- ibor_leg --------------------------------------------------------------


def test_ibor_leg_quarterly() -> None:
    schedule = Schedule.from_rule(
        effective_date=Date.from_ymd(1, Month.January, 2026),
        termination_date=Date.from_ymd(1, Month.January, 2027),
        tenor=Period(3, TimeUnit.Months),
        calendar=NullCalendar(),
        convention=BusinessDayConvention.Unadjusted,
        termination_date_convention=BusinessDayConvention.Unadjusted,
        rule=DateGeneration.Forward,
        end_of_month=False,
    )
    idx = _FlatIborIndex(0.035, fixing_days=0)
    leg = ibor_leg(
        schedule,
        idx,
        nominals=[100_000.0],
        payment_adjustment=BusinessDayConvention.Unadjusted,
        payment_calendar=NullCalendar(),
    )
    assert len(leg) == 4
    set_coupon_pricer(leg, BlackIborCouponPricer())
    # 4 quarters: 90, 91, 92, 92 days under Act/360 (Jan-Apr=90, Apr-Jul=91,
    # Jul-Oct=92, Oct-Jan=92). Sum = 365/360 * 0.035 * 100_000.
    total = sum(cf.amount() for cf in leg)
    expected = 100_000.0 * 0.035 * (90 + 91 + 92 + 92) / 360.0
    tolerance.tight(total, expected)


# --- OvernightIndexedCoupon -----------------------------------------------


def test_overnight_coupon_constant_fixing_compounds() -> None:
    """30-day window, daily fixing 4%, weekends-only calendar.

    Compound factor over the window equals
        prod(1 + 0.04 * (dt_i))
    where dt_i is the Act/360 day-count between consecutive business days.
    """
    d1 = Date.from_ymd(15, Month.January, 2026)
    d2 = Date.from_ymd(15, Month.February, 2026)
    idx = _FlatOvernightIndex(0.04)
    oic = OvernightIndexedCoupon(d2, 100_000.0, d1, d2, idx)
    # The internal pricer is attached by the ctor.
    rate = oic.rate()
    # Compound expectation: multiply (1 + 0.04 * dt_i) and divide by accrual_period.
    expected_compound = 1.0
    for dt in oic.dt():
        expected_compound *= 1.0 + 0.04 * dt
    expected_rate = (expected_compound - 1.0) / oic.accrual_period()
    tolerance.tight(rate, expected_rate)


def test_overnight_coupon_metadata_and_value_dates() -> None:
    d1 = Date.from_ymd(15, Month.January, 2026)
    d2 = Date.from_ymd(15, Month.February, 2026)
    idx = _FlatOvernightIndex(0.04)
    oic = OvernightIndexedCoupon(d2, 100_000.0, d1, d2, idx)
    assert oic.nominal() == 100_000.0
    assert len(oic.value_dates()) >= 2
    # Fixing dates and dt all have the same length (== n).
    assert len(oic.fixing_dates()) == oic.n()
    assert len(oic.dt()) == oic.n()
    # Each dt > 0
    for dt in oic.dt():
        assert dt > 0.0


def test_overnight_pricer_requires_overnight_coupon() -> None:
    d1 = Date.from_ymd(1, Month.January, 2026)
    d2 = Date.from_ymd(1, Month.April, 2026)
    idx = _FlatIborIndex(0.035, fixing_days=0)
    ic = IborCoupon(d2, 100_000.0, d1, d2, 0, idx)
    pricer = CompoundingOvernightIndexedCouponPricer()
    with pytest.raises(LibraryException, match="requires OvernightIndexedCoupon"):
        pricer.initialize(ic)


def test_overnight_leg() -> None:
    schedule = Schedule.from_rule(
        effective_date=Date.from_ymd(15, Month.January, 2026),
        termination_date=Date.from_ymd(15, Month.April, 2026),
        tenor=Period(1, TimeUnit.Months),
        calendar=WeekendsOnly(),
        convention=BusinessDayConvention.Following,
        termination_date_convention=BusinessDayConvention.Following,
        rule=DateGeneration.Forward,
        end_of_month=False,
    )
    idx = _FlatOvernightIndex(0.04)
    leg = overnight_leg(schedule, idx, nominals=[100_000.0])
    assert len(leg) == 3
    # Sanity: each coupon's amount > 0 at 4% rate
    for cf in leg:
        assert cf.amount() > 0.0


# --- Pricers ---------------------------------------------------------------


def test_pricer_caplet_floorlet_raise_on_base() -> None:
    p = IborCouponPricer()
    with pytest.raises(LibraryException, match="caplet pricing"):
        p.caplet_price(0.05)
    with pytest.raises(LibraryException, match="caplet pricing"):
        p.caplet_rate(0.05)
    with pytest.raises(LibraryException, match="floorlet pricing"):
        p.floorlet_price(0.03)
    with pytest.raises(LibraryException, match="floorlet pricing"):
        p.floorlet_rate(0.03)


def test_black_pricer_inherits_swaplet_behavior() -> None:
    d1 = Date.from_ymd(1, Month.January, 2026)
    d2 = Date.from_ymd(1, Month.April, 2026)
    idx = _FlatIborIndex(0.035, fixing_days=0)
    ic = IborCoupon(d2, 100_000.0, d1, d2, 0, idx)
    bp = BlackIborCouponPricer()
    ic.set_pricer(bp)
    # No cap/floor vol → equivalent to plain IborCouponPricer
    tolerance.tight(ic.rate(), 0.035)


def test_set_coupon_pricer_attaches_to_all_coupons() -> None:
    schedule = Schedule.from_rule(
        effective_date=Date.from_ymd(1, Month.January, 2026),
        termination_date=Date.from_ymd(1, Month.July, 2026),
        tenor=Period(3, TimeUnit.Months),
        calendar=NullCalendar(),
        convention=BusinessDayConvention.Unadjusted,
        termination_date_convention=BusinessDayConvention.Unadjusted,
        rule=DateGeneration.Forward,
        end_of_month=False,
    )
    idx = _FlatIborIndex(0.05, fixing_days=0)
    leg = ibor_leg(
        schedule,
        idx,
        nominals=[100_000.0],
        payment_adjustment=BusinessDayConvention.Unadjusted,
        payment_calendar=NullCalendar(),
    )
    pricer = IborCouponPricer()
    set_coupon_pricer(leg, pricer)
    for cf in leg:
        assert isinstance(cf, FloatingRateCoupon)
        assert cf.pricer() is pricer


def test_pricer_is_a_coupon_pricer() -> None:
    assert isinstance(IborCouponPricer(), CouponPricer)
    assert isinstance(BlackIborCouponPricer(), CouponPricer)
    assert isinstance(CompoundingOvernightIndexedCouponPricer(), CouponPricer)
