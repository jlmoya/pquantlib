"""Tests for the CashFlows aggregator + Duration enum."""

from __future__ import annotations

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.cashflows.duration import Duration
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.simple_cash_flow import SimpleCashFlow
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.interest_rate import InterestRate
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month

# --- Duration enum --------------------------------------------------------


def test_duration_enum_values_match_cpp() -> None:
    # C++ parity: ql/cashflows/duration.hpp:36 — Simple=0, Macaulay=1, Modified=2
    assert int(Duration.Simple) == 0
    assert int(Duration.Macaulay) == 1
    assert int(Duration.Modified) == 2


# --- CashFlows is namespace-only ------------------------------------------


def test_cash_flows_class_cannot_be_constructed() -> None:
    with pytest.raises(TypeError, match="namespace"):
        CashFlows()


# --- Mock flat discount curve ---------------------------------------------


class _FlatCompoundedCurve:
    """Flat (1+r)^(-tau) curve for testing CashFlows.npv / bps.

    Anchored at ``ref_date`` with Act365Fixed day count.
    """

    def __init__(self, ref_date: Date, rate: float = 0.05) -> None:
        self._ref = ref_date
        self._rate = rate
        self._dc = Actual365Fixed()

    def reference_date(self) -> Date:
        return self._ref

    def max_date(self) -> Date:
        # 100 years out — sufficient for L2-D testing.
        return Date.from_ymd(1, Month.January, 2126)

    def day_counter(self) -> DayCounter:
        return self._dc

    def discount(self, t: float | Date, extrapolate: bool = False) -> float:
        del extrapolate
        tau = (
            self._dc.year_fraction(self._ref, t)
            if isinstance(t, Date)
            else float(t)
        )
        return (1.0 + self._rate) ** (-tau)

    def zero_rate(self, arg: float | Date, extrapolate: bool = False) -> float:
        del arg, extrapolate
        return self._rate

    def forward_rate(
        self, t1: float | Date, t2: float | Date, extrapolate: bool = False
    ) -> float:
        del t1, t2, extrapolate
        return self._rate


# --- Reference fixture ----------------------------------------------------


@pytest.fixture(scope="module")
def ref_bond() -> dict[str, float]:
    return reference_reader.load("cluster/l2d")["bond_flat_curve"]


@pytest.fixture(scope="module")
def bond_leg() -> tuple[list[CashFlow], Date]:
    """Build the same 4-coupon + redemption leg as the C++ probe."""
    settle = Date.from_ymd(1, Month.January, 2026)
    coupon_dates = [
        (Date.from_ymd(1, Month.January, 2026), Date.from_ymd(1, Month.July, 2026)),
        (Date.from_ymd(1, Month.July, 2026), Date.from_ymd(1, Month.January, 2027)),
        (Date.from_ymd(1, Month.January, 2027), Date.from_ymd(1, Month.July, 2027)),
        (Date.from_ymd(1, Month.July, 2027), Date.from_ymd(1, Month.January, 2028)),
    ]
    leg: list[CashFlow] = []
    for start, end in coupon_dates:
        leg.append(
            FixedRateCoupon.from_rate(end, 100_000.0, 0.05, Actual360(), start, end)
        )
    # redemption
    leg.append(SimpleCashFlow(100_000.0, Date.from_ymd(1, Month.January, 2028)))
    return leg, settle


# --- npv via curve --------------------------------------------------------


def test_npv_curve_matches_cpp(
    ref_bond: dict[str, float], bond_leg: tuple[list[CashFlow], Date]
) -> None:
    leg, settle = bond_leg
    curve = _FlatCompoundedCurve(settle, 0.05)
    npv = CashFlows.npv_curve(leg, curve, settlement_date=settle)
    tolerance.tight(npv, ref_bond["npv"])


def test_npv_yield_matches_cpp(
    ref_bond: dict[str, float], bond_leg: tuple[list[CashFlow], Date]
) -> None:
    leg, settle = bond_leg
    y = InterestRate(0.05, Actual365Fixed(), Compounding.Compounded, Frequency.Annual)
    npv = CashFlows.npv_yield(leg, y, settlement_date=settle)
    tolerance.tight(npv, ref_bond["npv_at_ytm5"])


# --- bps -----------------------------------------------------------------


def test_bps_matches_cpp(
    ref_bond: dict[str, float], bond_leg: tuple[list[CashFlow], Date]
) -> None:
    leg, settle = bond_leg
    curve = _FlatCompoundedCurve(settle, 0.05)
    bps = CashFlows.bps(leg, curve, settlement_date=settle)
    tolerance.tight(bps, ref_bond["bps"])


# --- duration ------------------------------------------------------------


def test_simple_duration_matches_cpp(
    ref_bond: dict[str, float], bond_leg: tuple[list[CashFlow], Date]
) -> None:
    leg, settle = bond_leg
    y = InterestRate(0.05, Actual365Fixed(), Compounding.Compounded, Frequency.Annual)
    d = CashFlows.duration(leg, y, Duration.Simple, settlement_date=settle)
    tolerance.tight(d, ref_bond["simple_duration"])


def test_modified_duration_matches_cpp(
    ref_bond: dict[str, float], bond_leg: tuple[list[CashFlow], Date]
) -> None:
    leg, settle = bond_leg
    y = InterestRate(0.05, Actual365Fixed(), Compounding.Compounded, Frequency.Annual)
    d = CashFlows.duration(leg, y, Duration.Modified, settlement_date=settle)
    tolerance.tight(d, ref_bond["modified_duration"])


def test_macaulay_duration_matches_cpp(
    ref_bond: dict[str, float], bond_leg: tuple[list[CashFlow], Date]
) -> None:
    leg, settle = bond_leg
    y = InterestRate(0.05, Actual365Fixed(), Compounding.Compounded, Frequency.Annual)
    d = CashFlows.duration(leg, y, Duration.Macaulay, settlement_date=settle)
    tolerance.tight(d, ref_bond["macaulay_duration"])


def test_macaulay_duration_requires_compounded() -> None:
    leg: list[CashFlow] = [SimpleCashFlow(100.0, Date.from_ymd(1, Month.January, 2027))]
    y = InterestRate(0.05, Actual365Fixed(), Compounding.Simple, Frequency.Annual)
    from pquantlib.exceptions import LibraryException  # noqa: PLC0415

    with pytest.raises(LibraryException, match="compounded rate"):
        CashFlows.duration(
            leg, y, Duration.Macaulay, settlement_date=Date.from_ymd(1, Month.January, 2026)
        )


# --- convexity -----------------------------------------------------------


def test_convexity_matches_cpp(
    ref_bond: dict[str, float], bond_leg: tuple[list[CashFlow], Date]
) -> None:
    leg, settle = bond_leg
    y = InterestRate(0.05, Actual365Fixed(), Compounding.Compounded, Frequency.Annual)
    c = CashFlows.convexity(leg, y, settlement_date=settle)
    tolerance.tight(c, ref_bond["convexity"])


# --- IRR -----------------------------------------------------------------


def test_irr_recovers_yield_5pct(
    ref_bond: dict[str, float], bond_leg: tuple[list[CashFlow], Date]
) -> None:
    leg, settle = bond_leg
    # Use the curve NPV; IRR should recover ~5%.
    target_npv = ref_bond["npv_at_ytm5"]
    irr = CashFlows.irr(
        leg,
        target_npv,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
        settlement_date=settle,
        guess=0.04,
    )
    tolerance.loose(irr, ref_bond["irr"])
    # Also: irr should be ~ 5% within iterative tolerance.
    tolerance.loose(irr, 0.05)


# --- empty / edge cases ---------------------------------------------------


def test_empty_leg_npv_is_zero() -> None:
    curve = _FlatCompoundedCurve(Date.from_ymd(1, Month.January, 2026), 0.05)
    assert CashFlows.npv_curve([], curve) == 0.0


def test_empty_leg_bps_is_zero() -> None:
    curve = _FlatCompoundedCurve(Date.from_ymd(1, Month.January, 2026), 0.05)
    assert CashFlows.bps([], curve) == 0.0


def test_empty_leg_duration_is_zero() -> None:
    y = InterestRate(0.05, Actual365Fixed(), Compounding.Compounded, Frequency.Annual)
    assert (
        CashFlows.duration(
            [], y, Duration.Modified, settlement_date=Date.from_ymd(1, Month.January, 2026)
        )
        == 0.0
    )


def test_empty_leg_convexity_is_zero() -> None:
    y = InterestRate(0.05, Actual365Fixed(), Compounding.Compounded, Frequency.Annual)
    assert (
        CashFlows.convexity([], y, settlement_date=Date.from_ymd(1, Month.January, 2026))
        == 0.0
    )
