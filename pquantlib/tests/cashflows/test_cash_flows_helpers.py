"""Tests for the leg-walking helpers added to CashFlows in L3-B.

Covers ``start_date`` / ``maturity_date`` / ``is_expired`` /
``previous_cash_flow_date`` / ``next_cash_flow_date`` /
``accrued_amount`` / ``previous_coupon_rate`` / ``next_coupon_rate``.
"""

from __future__ import annotations

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.simple_cash_flow import SimpleCashFlow
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.interest_rate import InterestRate
from pquantlib.testing import tolerance
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month


def _ir() -> InterestRate:
    return InterestRate(
        0.05, Thirty360(Convention.BondBasis), Compounding.Simple, Frequency.Annual,
    )


def _leg() -> list[CashFlow]:
    """3 annual coupons + final redemption (Jan 15, 2025 -> 2028)."""
    dates = [
        Date.from_ymd(15, Month.January, 2025),
        Date.from_ymd(15, Month.January, 2026),
        Date.from_ymd(15, Month.January, 2027),
        Date.from_ymd(15, Month.January, 2028),
    ]
    leg: list[CashFlow] = []
    for i in range(3):
        leg.append(
            FixedRateCoupon(dates[i + 1], 100.0, _ir(), dates[i], dates[i + 1])
        )
    leg.append(SimpleCashFlow(100.0, dates[-1]))
    return leg


def test_start_date_returns_earliest_accrual() -> None:
    leg = _leg()
    assert CashFlows.start_date(leg) == Date.from_ymd(15, Month.January, 2025)


def test_start_date_empty_leg_raises() -> None:
    with pytest.raises(LibraryException, match="empty leg"):
        CashFlows.start_date([])


def test_maturity_date_returns_latest_end() -> None:
    leg = _leg()
    assert CashFlows.maturity_date(leg) == Date.from_ymd(15, Month.January, 2028)


def test_is_expired_before_first_cf() -> None:
    leg = _leg()
    settle = Date.from_ymd(1, Month.January, 2025)
    assert CashFlows.is_expired(leg, None, settle) is False


def test_is_expired_after_last_cf() -> None:
    leg = _leg()
    settle = Date.from_ymd(1, Month.January, 2030)
    assert CashFlows.is_expired(leg, None, settle) is True


def test_is_expired_empty_leg() -> None:
    assert CashFlows.is_expired([], None, Date.from_ymd(1, Month.January, 2025)) is True


def test_next_cash_flow_date_mid_period() -> None:
    leg = _leg()
    # Mid-2025 → next cf is the Jan 15, 2026 coupon.
    settle = Date.from_ymd(1, Month.July, 2025)
    assert CashFlows.next_cash_flow_date(leg, None, settle) == Date.from_ymd(
        15, Month.January, 2026
    )


def test_previous_cash_flow_date_mid_period() -> None:
    leg = _leg()
    settle = Date.from_ymd(1, Month.July, 2026)
    # First coupon (2026) has occurred.
    assert CashFlows.previous_cash_flow_date(leg, None, settle) == Date.from_ymd(
        15, Month.January, 2026
    )


def test_previous_cash_flow_date_returns_null_when_none() -> None:
    leg = _leg()
    settle = Date.from_ymd(1, Month.January, 2025)
    # Null Date — has serial 0.
    assert CashFlows.previous_cash_flow_date(leg, None, settle).serial_number() == 0


def test_next_cash_flow_date_returns_null_after_last() -> None:
    leg = _leg()
    settle = Date.from_ymd(15, Month.January, 2030)
    assert CashFlows.next_cash_flow_date(leg, None, settle).serial_number() == 0


def test_accrued_amount_mid_first_period() -> None:
    """At mid-2025, accrual_period covers Jan 15 → Jul 15 (= 6m / 12)."""
    leg = _leg()
    mid = Date.from_ymd(15, Month.July, 2025)
    # FixedRateCoupon.accrued_amount = nominal * ((1 + rate*tau)*tau / period_period - 1)... actually:
    # FixedRateCoupon.accrued_amount(d) returns nominal * (compoundFactor(start,d) - 1)
    # for a Simple/Annual rate at 5%, accrued = 100 * 0.05 * 0.5 = 2.5
    accrued = CashFlows.accrued_amount(leg, False, mid)
    tolerance.tight(accrued, 2.5)


def test_next_coupon_rate_returns_blended_rate() -> None:
    """Single coupon at the next-date → just its rate (0.05)."""
    leg = _leg()
    mid = Date.from_ymd(1, Month.July, 2025)
    tolerance.tight(CashFlows.next_coupon_rate(leg, None, mid), 0.05)


def test_previous_coupon_rate_zero_when_no_prior() -> None:
    leg = _leg()
    settle = Date.from_ymd(1, Month.January, 2025)
    tolerance.tight(CashFlows.previous_coupon_rate(leg, None, settle), 0.0)
