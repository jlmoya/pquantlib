"""Tests for pquantlib.cashflows.cash_flow + simple_cash_flow + coupon abstract."""

from __future__ import annotations

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.coupon import Coupon
from pquantlib.cashflows.simple_cash_flow import (
    AmortizingPayment,
    Redemption,
    SimpleCashFlow,
)
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def ref_simple() -> dict[str, float]:
    return reference_reader.load("cluster/l2d")["simple_cash_flow"]


# --- CashFlow base abstract -------------------------------------------------


def test_cash_flow_is_abstract() -> None:
    with pytest.raises(TypeError):
        CashFlow()  # type: ignore[abstract]


# --- SimpleCashFlow ---------------------------------------------------------


def test_simple_cash_flow_round_trip(ref_simple: dict[str, float]) -> None:
    d = Date.from_ymd(1, Month.July, 2026)
    scf = SimpleCashFlow(1234.56, d)
    tolerance.tight(scf.amount(), ref_simple["amount"])
    assert scf.date() == d
    # date_serial comparison
    assert scf.date().serial_number() == int(ref_simple["date_serial"])


def test_simple_cash_flow_subclasses_keep_amount_and_date() -> None:
    d = Date.from_ymd(1, Month.July, 2026)
    red = Redemption(100.0, d)
    amo = AmortizingPayment(50.0, d)
    assert red.amount() == 100.0
    assert amo.amount() == 50.0
    assert red.date() == d
    assert amo.date() == d
    # Subclass identity preserved.
    assert isinstance(red, SimpleCashFlow)
    assert isinstance(amo, SimpleCashFlow)


def test_has_occurred_default_no_ref_date_returns_false() -> None:
    d = Date.from_ymd(1, Month.July, 2026)
    scf = SimpleCashFlow(100.0, d)
    # No reference -> False (cannot determine without explicit ref).
    assert scf.has_occurred() is False


def test_has_occurred_before_after_equal_ref_date() -> None:
    d = Date.from_ymd(1, Month.July, 2026)
    scf = SimpleCashFlow(100.0, d)
    earlier = Date.from_ymd(1, Month.June, 2026)
    later = Date.from_ymd(1, Month.August, 2026)
    assert scf.has_occurred(earlier) is False
    assert scf.has_occurred(later) is True
    # On the date: default (include_ref_date=None -> treated as False)
    # means "event has occurred on its own date".
    assert scf.has_occurred(d) is True
    # Explicit include_ref_date=True flips the result (mirrors C++).
    assert scf.has_occurred(d, include_ref_date=True) is False


def test_trading_ex_coupon_returns_false_with_no_ex_coupon_date() -> None:
    d = Date.from_ymd(1, Month.July, 2026)
    scf = SimpleCashFlow(100.0, d)
    # Default ex_coupon_date is null Date -> always False.
    assert scf.trading_ex_coupon(Date.from_ymd(1, Month.June, 2026)) is False


def test_cash_flow_is_observable() -> None:
    d = Date.from_ymd(1, Month.July, 2026)
    scf = SimpleCashFlow(100.0, d)

    notified: list[int] = []

    class _O:
        def update(self) -> None:
            notified.append(1)

    o = _O()
    scf.register_with(o)
    scf.notify_observers()
    assert notified == [1]


# --- Coupon abstract -------------------------------------------------------


class _FixedRateProbeCoupon(Coupon):
    """Minimal concrete Coupon for testing the abstract base.

    Pays nominal * rate * accrual_period; rate / day_counter / accrued_amount
    are returned as constants.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        rate: float,
        dc: DayCounter,
        start: Date,
        end: Date,
    ) -> None:
        super().__init__(payment_date, nominal, start, end)
        self._rate = rate
        self._dc = dc

    def rate(self) -> float:
        return self._rate

    def day_counter(self) -> DayCounter:
        return self._dc

    def amount(self) -> float:
        return self._nominal * self._rate * self.accrual_period()

    def accrued_amount(self, d: Date) -> float:
        return self._nominal * self._rate * self.accrued_period(d)


def test_coupon_accrual_period_and_days() -> None:
    d1 = Date.from_ymd(1, Month.January, 2026)
    d2 = Date.from_ymd(1, Month.July, 2026)
    cp = _FixedRateProbeCoupon(d2, 100000.0, 0.05, Actual360(), d1, d2)
    # 181 days / 360 = 0.502777...
    assert cp.accrual_days() == 181
    tolerance.tight(cp.accrual_period(), 181.0 / 360.0)
    tolerance.tight(cp.accrual_period(), 181.0 / 360.0)  # memoized roundtrip


def test_coupon_nominal_and_dates_pass_through() -> None:
    d1 = Date.from_ymd(1, Month.January, 2026)
    d2 = Date.from_ymd(1, Month.July, 2026)
    cp = _FixedRateProbeCoupon(d2, 100_000.0, 0.05, Actual360(), d1, d2)
    assert cp.nominal() == 100_000.0
    assert cp.accrual_start_date() == d1
    assert cp.accrual_end_date() == d2
    # ref period falls back to accrual period
    assert cp.reference_period_start() == d1
    assert cp.reference_period_end() == d2
    # date() == payment_date
    assert cp.date() == d2


def test_coupon_accrued_period_zero_outside_window() -> None:
    d1 = Date.from_ymd(1, Month.January, 2026)
    d2 = Date.from_ymd(1, Month.July, 2026)
    cp = _FixedRateProbeCoupon(d2, 100_000.0, 0.05, Actual360(), d1, d2)
    # Before accrual_start_date
    earlier = Date.from_ymd(1, Month.December, 2025)
    assert cp.accrued_period(earlier) == 0.0
    # After payment_date (also d2 here)
    later = Date.from_ymd(2, Month.July, 2026)
    assert cp.accrued_period(later) == 0.0
    # Mid-period
    mid = Date.from_ymd(1, Month.April, 2026)
    expected_days = mid - d1
    tolerance.tight(cp.accrued_period(mid), float(expected_days) / 360.0)


def test_coupon_is_abstract() -> None:
    # Cannot instantiate Coupon directly.
    with pytest.raises(TypeError):
        Coupon(  # type: ignore[abstract]
            Date.from_ymd(1, Month.July, 2026),
            100.0,
            Date.from_ymd(1, Month.January, 2026),
            Date.from_ymd(1, Month.July, 2026),
        )
