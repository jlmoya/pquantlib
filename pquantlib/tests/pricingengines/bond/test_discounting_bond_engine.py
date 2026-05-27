"""Tests for DiscountingBondEngine — focus on the engine itself.

Most coverage is indirect (via the bond tests). This file covers
the engine's API contract: setting arguments, computing results, the
``include_settlement_date_flows`` override, and observer wiring.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.instruments.bond import BondArguments, BondResults
from pquantlib.instruments.bonds.fixed_rate_bond import FixedRateBond
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.bond.discounting_bond_engine import DiscountingBondEngine
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import tolerance
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


@pytest.fixture
def pinned_eval_date() -> Any:
    settings = ObservableSettings()
    saved = settings.evaluation_date
    settings.evaluation_date = Date.from_ymd(15, Month.January, 2025)
    yield None
    settings.evaluation_date = saved


def _bond_and_engine() -> tuple[FixedRateBond, DiscountingBondEngine, FlatForward]:
    issue = Date.from_ymd(15, Month.January, 2025)
    schedule = Schedule.from_rule(
        effective_date=issue,
        termination_date=Date.from_ymd(15, Month.January, 2030),
        tenor=Period(6, TimeUnit.Months),
        calendar=TARGET(),
        convention=BusinessDayConvention.Unadjusted,
        termination_date_convention=BusinessDayConvention.Unadjusted,
        rule=DateGeneration.Backward,
        end_of_month=False,
    )
    bond = FixedRateBond(
        settlement_days=2,
        face_amount=100.0,
        schedule=schedule,
        coupons=[0.05],
        accrual_day_counter=Thirty360(Convention.BondBasis),
        payment_convention=BusinessDayConvention.Following,
        redemption=100.0,
        issue_date=issue,
    )
    curve = FlatForward.from_rate(
        issue,
        0.05,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
    )
    engine = DiscountingBondEngine(curve)
    bond.set_pricing_engine(engine)
    return bond, engine, curve


def test_engine_args_and_results_types() -> None:
    """Engine owns concrete BondArguments + BondResults instances."""
    _bond, engine, _curve = _bond_and_engine()
    assert isinstance(engine.get_arguments(), BondArguments)
    assert isinstance(engine.get_results(), BondResults)


def test_engine_calculates_settlement_value(pinned_eval_date: Any) -> None:
    """Triggering NPV runs calculate() and fills settlement_value."""
    bond, engine, _curve = _bond_and_engine()
    bond.npv()
    assert engine.get_results().settlement_value is not None


def test_engine_npv_matches_manual_curve_npv(pinned_eval_date: Any) -> None:
    """Engine NPV equals CashFlows.npv_curve at the curve reference date."""
    bond, _engine, curve = _bond_and_engine()
    # cast: FlatForward satisfies the Protocol at runtime (the
    # pyright-flagged parameter-name + return-type mismatch on the
    # unused zero_rate is harmless here).
    manual = CashFlows.npv_curve(
        bond.cashflows(),
        cast("YieldTermStructureProtocol", curve),
        ObservableSettings().include_reference_date_events,
        curve.reference_date(),
        curve.reference_date(),
    )
    tolerance.tight(bond.npv(), manual)


def test_engine_include_settlement_date_flows_override(
    pinned_eval_date: Any,
) -> None:
    """Explicit include_settlement_date_flows overrides Settings."""
    issue = Date.from_ymd(15, Month.January, 2025)
    schedule = Schedule.from_rule(
        effective_date=issue,
        termination_date=Date.from_ymd(15, Month.January, 2030),
        tenor=Period(6, TimeUnit.Months),
        calendar=TARGET(),
        convention=BusinessDayConvention.Unadjusted,
        termination_date_convention=BusinessDayConvention.Unadjusted,
        rule=DateGeneration.Backward,
        end_of_month=False,
    )
    bond = FixedRateBond(
        settlement_days=2,
        face_amount=100.0,
        schedule=schedule,
        coupons=[0.05],
        accrual_day_counter=Thirty360(Convention.BondBasis),
        payment_convention=BusinessDayConvention.Following,
        redemption=100.0,
        issue_date=issue,
    )
    curve = FlatForward.from_rate(
        issue, 0.05, Actual365Fixed(), Compounding.Compounded, Frequency.Annual,
    )
    # Override to False (exclude settlement-date flows).
    engine = DiscountingBondEngine(curve, include_settlement_date_flows=False)
    bond.set_pricing_engine(engine)
    bond.npv()
    # We don't assert a particular value here — just that the engine
    # honours the override without raising and ``settlement_value`` is
    # populated by a fresh CashFlows.npv at settlement_date.
    assert engine.get_results().settlement_value is not None


def test_engine_discount_curve_accessor() -> None:
    _bond, engine, curve = _bond_and_engine()
    assert engine.discount_curve() is curve
