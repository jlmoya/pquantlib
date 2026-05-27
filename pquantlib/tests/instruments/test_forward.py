"""Tests for Forward + ForwardTypePayoff.

Reference values come from ``cluster/l3e.json`` emitted by
``migration-harness/cpp/probes/cluster_l3e/probe.cpp``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.forward import Forward, ForwardTypePayoff
from pquantlib.payoffs import Payoff
from pquantlib.position import PositionType
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from tests.indexes._mock_curves import FlatForwardMock


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l3e")


# --- ForwardTypePayoff -----------------------------------------------------


def test_forward_type_payoff_name(ref: dict[str, Any]) -> None:
    payoff = ForwardTypePayoff(PositionType.Long, 100.0)
    assert payoff.name() == ref["forward_type_payoff"]["name"]


def test_forward_type_payoff_description() -> None:
    payoff = ForwardTypePayoff(PositionType.Long, 100.0)
    # C++ description: "Forward, 100 strike" — Python uses default float repr
    # so we check membership rather than exact match.
    assert "Forward" in payoff.description()
    assert "100" in payoff.description()


def test_forward_type_payoff_strike_and_type() -> None:
    payoff = ForwardTypePayoff(PositionType.Long, 100.0)
    assert payoff.strike() == 100.0
    assert payoff.forward_type() == PositionType.Long


def test_forward_type_payoff_long(ref: dict[str, Any]) -> None:
    payoff = ForwardTypePayoff(PositionType.Long, 100.0)
    expected = ref["forward_type_payoff"]
    # EXACT tier — pure arithmetic.
    exact(payoff(120.0), float(expected["long_at_120"]))
    exact(payoff(100.0), float(expected["long_at_100"]))
    exact(payoff(80.0), float(expected["long_at_80"]))


def test_forward_type_payoff_short(ref: dict[str, Any]) -> None:
    payoff = ForwardTypePayoff(PositionType.Short, 100.0)
    expected = ref["forward_type_payoff"]
    exact(payoff(120.0), float(expected["short_at_120"]))
    exact(payoff(100.0), float(expected["short_at_100"]))
    exact(payoff(80.0), float(expected["short_at_80"]))


def test_forward_type_payoff_negative_strike_rejected() -> None:
    with pytest.raises(LibraryException, match="negative strike"):
        ForwardTypePayoff(PositionType.Long, -1.0)


# --- Forward abstract base -------------------------------------------------


class _SimpleForward(Forward):
    """Concrete Forward stub for testing the base class lifecycle."""

    def __init__(
        self,
        spot: float,
        income: float,
        payoff: Payoff,
        value_date: Date,
        maturity_date: Date,
        discount_curve: YieldTermStructureProtocol,
    ) -> None:
        super().__init__(
            day_counter=Actual360(),
            calendar=TARGET(),
            business_day_convention=BusinessDayConvention.Following,
            settlement_days=2,
            payoff=payoff,
            value_date=value_date,
            maturity_date=maturity_date,
            discount_curve=discount_curve,
        )
        self._spot: float = spot
        self._income: float = income

    def spot_value(self) -> float:
        return self._spot

    def spot_income(
        self, income_discount_curve: YieldTermStructureProtocol | None
    ) -> float:
        del income_discount_curve
        return self._income

    def _perform_calculations(self) -> None:
        # Subclass MUST set these before chaining to base.
        self._underlying_spot_value = self.spot_value()
        self._underlying_income = self.spot_income(self._discount_curve)
        super()._perform_calculations()


def test_forward_cannot_instantiate_abstract() -> None:
    with pytest.raises(TypeError):
        Forward(  # type: ignore[abstract]
            day_counter=Actual360(),
            calendar=TARGET(),
            business_day_convention=BusinessDayConvention.Following,
            settlement_days=2,
            payoff=ForwardTypePayoff(PositionType.Long, 100.0),
            value_date=Date.from_ymd(17, Month.January, 2024),
            maturity_date=Date.from_ymd(17, Month.January, 2025),
        )


def test_forward_npv_long() -> None:
    """End-to-end forward NPV: 100 spot, 0 income, 100 strike, FlatForward(0%).

    Forward value = (spot - income) / discount(T) = 100 / 1 = 100.
    Payoff(forward_value=100) = 100 - strike = 0; NPV = 0.
    """
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.0, Actual360())
    value_date = Date.from_ymd(19, Month.January, 2024)
    maturity = Date.from_ymd(17, Month.January, 2025)
    fwd = _SimpleForward(
        spot=100.0,
        income=0.0,
        payoff=ForwardTypePayoff(PositionType.Long, 100.0),
        value_date=value_date,
        maturity_date=maturity,
        discount_curve=ts,
    )
    exact(fwd.npv(), 0.0)


def test_forward_npv_long_with_drift() -> None:
    """100 spot, 0 income, strike 95: long fwd profit at maturity."""
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    value_date = Date.from_ymd(19, Month.January, 2024)
    maturity = Date.from_ymd(17, Month.January, 2025)
    fwd = _SimpleForward(
        spot=100.0,
        income=0.0,
        payoff=ForwardTypePayoff(PositionType.Long, 95.0),
        value_date=value_date,
        maturity_date=maturity,
        discount_curve=ts,
    )
    # forward_value = 100 / discount(T)
    # NPV = (forward_value - 95) * discount(T) = 100 - 95 * discount(T)
    discount_t = ts.discount(maturity)
    expected_npv = 100.0 - 95.0 * discount_t
    # TIGHT tier — the recompute path triggers another exp() call inside
    # the curve which may differ by 1 ULP from the direct one above.
    tight(fwd.npv(), expected_npv)


def test_forward_value_date_inspector() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    value_date = Date.from_ymd(19, Month.January, 2024)
    maturity = Date.from_ymd(17, Month.January, 2025)
    fwd = _SimpleForward(
        spot=100.0,
        income=0.0,
        payoff=ForwardTypePayoff(PositionType.Long, 100.0),
        value_date=value_date,
        maturity_date=maturity,
        discount_curve=ts,
    )
    assert fwd.value_date() == value_date
    # Maturity may have been adjusted via Following — Jan 17, 2025 is a Friday
    # so stays the same.
    assert fwd.maturity_date() == maturity
    assert fwd.discount_curve() is ts
    assert fwd.day_counter().name() == "Actual/360"
