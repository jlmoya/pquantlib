"""Tests for AmortizingFixedRateBond + helper utilities."""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.instruments.bonds.amortizing_fixed_rate_bond import (
    AmortizingFixedRateBond,
    sinking_notionals,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.bond.discounting_bond_engine import DiscountingBondEngine
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l3b")["amortizing_fixed_rate_bond"]


@pytest.fixture
def pinned_eval_date() -> Any:
    settings = ObservableSettings()
    saved = settings.evaluation_date
    settings.evaluation_date = Date.from_ymd(15, Month.January, 2025)
    yield None
    settings.evaluation_date = saved


def _amortising() -> AmortizingFixedRateBond:
    """4y annual 5%, linear notionals [100, 75, 50, 25]."""
    issue = Date.from_ymd(15, Month.January, 2025)
    maturity = Date.from_ymd(15, Month.January, 2029)
    schedule = Schedule.from_rule(
        effective_date=issue,
        termination_date=maturity,
        tenor=Period(1, TimeUnit.Years),
        calendar=TARGET(),
        convention=BusinessDayConvention.Unadjusted,
        termination_date_convention=BusinessDayConvention.Unadjusted,
        rule=DateGeneration.Backward,
        end_of_month=False,
    )
    return AmortizingFixedRateBond(
        settlement_days=2,
        notionals=[100.0, 75.0, 50.0, 25.0],
        schedule=schedule,
        coupons=[0.05],
        accrual_day_counter=Thirty360(Convention.BondBasis),
        payment_convention=BusinessDayConvention.Following,
        issue_date=issue,
        redemptions=[100.0],
    )


def test_amortising_structure(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _amortising()
    # 4 coupons + 3 AmortizingPayments + 1 final Redemption = 8.
    assert len(bond.cashflows()) == ref["n_cashflows"]


def test_amortising_settle_serial(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _amortising()
    assert bond.settlement_date().serial_number() == ref["settle_serial"]


def test_amortising_notional_schedule(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _amortising()
    assert bond.notional(bond.settlement_date()) == ref["notional_at_settle"]
    assert bond.notional(Date.from_ymd(15, Month.January, 2026)) == ref["notional_y1"]
    assert bond.notional(Date.from_ymd(15, Month.January, 2027)) == ref["notional_y2"]
    assert bond.notional(Date.from_ymd(15, Month.January, 2028)) == ref["notional_y3"]


def test_amortising_npv(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _amortising()
    curve = FlatForward.from_rate(
        Date.from_ymd(15, Month.January, 2025),
        0.05,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
    )
    bond.set_pricing_engine(DiscountingBondEngine(curve))
    tolerance.tight(bond.npv(), ref["npv"])


def test_amortising_clean_price(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _amortising()
    curve = FlatForward.from_rate(
        Date.from_ymd(15, Month.January, 2025),
        0.05,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
    )
    bond.set_pricing_engine(DiscountingBondEngine(curve))
    tolerance.tight(bond.clean_price(), ref["clean_price"])


def test_amortising_dirty_price(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _amortising()
    curve = FlatForward.from_rate(
        Date.from_ymd(15, Month.January, 2025),
        0.05,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
    )
    bond.set_pricing_engine(DiscountingBondEngine(curve))
    tolerance.tight(bond.dirty_price(), ref["dirty_price"])


# --- sinking_notionals helper -------------------------------------------


def test_sinking_notionals_french_amortisation() -> None:
    """4y annual @ 5% → 4 French-amortisation notionals.

    Sanity-checked against the C++ logic: the schedule monotonically
    decreases to zero, and the first entry equals the initial notional.
    """
    notionals = sinking_notionals(
        Period(4, TimeUnit.Years),
        Frequency.Annual,
        0.05,
        100.0,
    )
    # n_periods + 1 entries
    assert len(notionals) == 5
    assert notionals[0] == 100.0
    assert notionals[-1] == 0.0
    # Monotonically non-increasing
    for i in range(1, len(notionals)):
        assert notionals[i] <= notionals[i - 1]


def test_sinking_notionals_zero_coupon_uses_linear() -> None:
    """At coupon < 1e-12 the helper falls back to straight-line."""
    notionals = sinking_notionals(
        Period(4, TimeUnit.Years),
        Frequency.Annual,
        0.0,
        100.0,
    )
    # 100, 75, 50, 25, 0 (linear)
    tolerance.tight(notionals[1], 75.0)
    tolerance.tight(notionals[2], 50.0)
    tolerance.tight(notionals[3], 25.0)
    assert notionals[-1] == 0.0
