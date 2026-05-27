"""Tests for FixedRateCoupon + fixed_rate_leg."""

from __future__ import annotations

import pytest

from pquantlib.cashflows.compounding import Compounding
from pquantlib.cashflows.coupon import Coupon
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.fixed_rate_leg import fixed_rate_leg
from pquantlib.cashflows.interest_rate import InterestRate
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def ref_frc() -> dict[str, float]:
    return reference_reader.load("cluster/l2d")["fixed_rate_coupon"]


# --- FixedRateCoupon -------------------------------------------------------


def test_fixed_rate_coupon_amount_matches_cpp(ref_frc: dict[str, float]) -> None:
    d1 = Date.from_ymd(1, Month.January, 2026)
    d2 = Date.from_ymd(1, Month.July, 2026)
    ir = InterestRate(0.05, Actual360(), Compounding.Simple, Frequency.Annual)
    frc = FixedRateCoupon(d2, 100_000.0, ir, d1, d2)
    tolerance.tight(frc.amount(), ref_frc["amount"])
    # And via the from_rate factory
    frc2 = FixedRateCoupon.from_rate(d2, 100_000.0, 0.05, Actual360(), d1, d2)
    tolerance.tight(frc2.amount(), ref_frc["amount"])


def test_fixed_rate_coupon_metadata(ref_frc: dict[str, float]) -> None:
    d1 = Date.from_ymd(1, Month.January, 2026)
    d2 = Date.from_ymd(1, Month.July, 2026)
    ir = InterestRate(0.05, Actual360(), Compounding.Simple, Frequency.Annual)
    frc = FixedRateCoupon(d2, 100_000.0, ir, d1, d2)
    assert frc.nominal() == 100_000.0
    assert frc.rate() == 0.05
    assert frc.day_counter() == Actual360()
    assert frc.accrual_days() == int(ref_frc["accrual_days"])
    tolerance.tight(frc.accrual_period(), ref_frc["accrual_period"])
    assert frc.date().serial_number() == int(ref_frc["payment_date_serial"])


def test_fixed_rate_coupon_is_a_coupon() -> None:
    d1 = Date.from_ymd(1, Month.January, 2026)
    d2 = Date.from_ymd(1, Month.July, 2026)
    frc = FixedRateCoupon.from_rate(d2, 100_000.0, 0.05, Actual360(), d1, d2)
    assert isinstance(frc, Coupon)


def test_accrued_amount_mid_period_equals_simple_proportion() -> None:
    d1 = Date.from_ymd(1, Month.January, 2026)
    d2 = Date.from_ymd(1, Month.July, 2026)
    mid = Date.from_ymd(1, Month.April, 2026)
    frc = FixedRateCoupon.from_rate(d2, 100_000.0, 0.05, Actual360(), d1, d2)
    # Simple/Annual compound_factor over (d1, mid) = 1 + 0.05 * (90/360).
    # → accrued_amount = 100,000 * 0.05 * (90/360) = 1,250.
    tolerance.tight(frc.accrued_amount(mid), 100_000.0 * 0.05 * (90.0 / 360.0))


def test_accrued_amount_zero_outside_window() -> None:
    d1 = Date.from_ymd(1, Month.January, 2026)
    d2 = Date.from_ymd(1, Month.July, 2026)
    frc = FixedRateCoupon.from_rate(d2, 100_000.0, 0.05, Actual360(), d1, d2)
    earlier = Date.from_ymd(1, Month.December, 2025)
    later = Date.from_ymd(2, Month.July, 2026)
    assert frc.accrued_amount(earlier) == 0.0
    assert frc.accrued_amount(later) == 0.0


# --- fixed_rate_leg --------------------------------------------------------


def test_fixed_rate_leg_4_coupons_semiannual() -> None:
    """2y semiannual bond — 4 coupons of 2500 each."""
    schedule = Schedule.from_rule(
        effective_date=Date.from_ymd(1, Month.January, 2026),
        termination_date=Date.from_ymd(1, Month.January, 2028),
        tenor=Period(6, TimeUnit.Months),
        calendar=NullCalendar(),
        convention=BusinessDayConvention.Unadjusted,
        termination_date_convention=BusinessDayConvention.Unadjusted,
        rule=DateGeneration.Forward,
        end_of_month=False,
    )

    leg = fixed_rate_leg(
        schedule,
        nominals=[100_000.0],
        rates=[0.05],
        day_counter=Actual360(),
        compounding=Compounding.Simple,
        frequency=Frequency.Annual,
        payment_adjustment=BusinessDayConvention.Unadjusted,
        payment_calendar=NullCalendar(),
    )

    assert len(leg) == 4
    # First period: Jan 1 -> Jul 1 = 181 days; 100_000 * 0.05 * 181/360 = 2513.888...
    first = leg[0]
    assert isinstance(first, FixedRateCoupon)
    tolerance.tight(first.amount(), 100_000.0 * 0.05 * (181.0 / 360.0))
    # Sum amounts == sum of (notional * rate * days_in_period / 360)
    total = sum(cf.amount() for cf in leg)
    # Day counts: 181, 184, 181, 184 = 730/360 * 5000 = 10138.888...
    expected = 100_000.0 * 0.05 * (181 + 184 + 181 + 184) / 360.0
    tolerance.tight(total, expected)


def test_fixed_rate_leg_per_period_notionals_and_rates() -> None:
    """Each period gets its own rate + nominal; short rates/nominals fall back to last."""
    schedule = Schedule.from_rule(
        effective_date=Date.from_ymd(1, Month.January, 2026),
        termination_date=Date.from_ymd(1, Month.January, 2027),
        tenor=Period(6, TimeUnit.Months),
        calendar=NullCalendar(),
        convention=BusinessDayConvention.Unadjusted,
        termination_date_convention=BusinessDayConvention.Unadjusted,
        rule=DateGeneration.Forward,
        end_of_month=False,
    )
    leg = fixed_rate_leg(
        schedule,
        nominals=[100_000.0, 200_000.0],
        rates=[0.04, 0.06],
        day_counter=Actual360(),
        payment_adjustment=BusinessDayConvention.Unadjusted,
        payment_calendar=NullCalendar(),
    )
    assert len(leg) == 2
    # First: 100_000 * 0.04 * 181/360
    tolerance.tight(leg[0].amount(), 100_000.0 * 0.04 * (181.0 / 360.0))
    # Second: 200_000 * 0.06 * 184/360
    tolerance.tight(leg[1].amount(), 200_000.0 * 0.06 * (184.0 / 360.0))


def test_fixed_rate_leg_empty_inputs_raise() -> None:
    schedule = Schedule.from_rule(
        effective_date=Date.from_ymd(1, Month.January, 2026),
        termination_date=Date.from_ymd(1, Month.July, 2026),
        tenor=Period(6, TimeUnit.Months),
        calendar=NullCalendar(),
        convention=BusinessDayConvention.Unadjusted,
        termination_date_convention=BusinessDayConvention.Unadjusted,
        rule=DateGeneration.Forward,
        end_of_month=False,
    )
    with pytest.raises(LibraryException, match="no coupon rates"):
        fixed_rate_leg(
            schedule, nominals=[100_000.0], rates=[], day_counter=Actual360()
        )
    with pytest.raises(LibraryException, match="no notional"):
        fixed_rate_leg(schedule, nominals=[], rates=[0.05], day_counter=Actual360())


def test_fixed_rate_leg_with_interest_rate_objects() -> None:
    """Passing pre-built InterestRate ignores day_counter/compounding/frequency."""
    schedule = Schedule.from_rule(
        effective_date=Date.from_ymd(1, Month.January, 2026),
        termination_date=Date.from_ymd(1, Month.July, 2026),
        tenor=Period(6, TimeUnit.Months),
        calendar=NullCalendar(),
        convention=BusinessDayConvention.Unadjusted,
        termination_date_convention=BusinessDayConvention.Unadjusted,
        rule=DateGeneration.Forward,
        end_of_month=False,
    )
    ir = InterestRate(0.05, Actual360(), Compounding.Simple, Frequency.Annual)
    leg = fixed_rate_leg(
        schedule,
        nominals=[100_000.0],
        rates=[ir],
        payment_adjustment=BusinessDayConvention.Unadjusted,
        payment_calendar=NullCalendar(),
    )
    assert len(leg) == 1
    tolerance.tight(leg[0].amount(), 100_000.0 * 0.05 * (181.0 / 360.0))
