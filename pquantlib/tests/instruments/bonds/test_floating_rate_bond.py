"""Tests for FloatingRateBond + DiscountingBondEngine."""

from __future__ import annotations

from typing import Any, cast

import pytest

from pquantlib.cashflows.coupon_pricer import BlackIborCouponPricer, set_coupon_pricer
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.bonds.floating_rate_bond import FloatingRateBond
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.bond.discounting_bond_engine import DiscountingBondEngine
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
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
    return reference_reader.load("cluster/l3b")["floating_rate_bond"]


@pytest.fixture
def pinned_eval_date() -> Any:
    """Same eval-date pinning as the probe (Dec 1, 2024)."""
    settings = ObservableSettings()
    saved = settings.evaluation_date
    settings.evaluation_date = Date.from_ymd(1, Month.December, 2024)
    yield None
    settings.evaluation_date = saved


def _build_floater() -> FloatingRateBond:
    issue = Date.from_ymd(15, Month.January, 2025)
    maturity = Date.from_ymd(15, Month.January, 2030)
    forecast = FlatForward.from_rate(
        Date.from_ymd(1, Month.December, 2024),
        0.035,
        Actual360(),
        Compounding.Simple,
        Frequency.Annual,
    )
    idx = Euribor.six_months(cast("YieldTermStructureProtocol", forecast))
    schedule = Schedule.from_rule(
        effective_date=issue,
        termination_date=maturity,
        tenor=Period(6, TimeUnit.Months),
        calendar=TARGET(),
        convention=BusinessDayConvention.Unadjusted,
        termination_date_convention=BusinessDayConvention.Unadjusted,
        rule=DateGeneration.Backward,
        end_of_month=False,
    )
    return FloatingRateBond(
        settlement_days=2,
        face_amount=100.0,
        schedule=schedule,
        ibor_index=idx,
        accrual_day_counter=Actual360(),
        payment_convention=BusinessDayConvention.Following,
        fixing_days=2,
        gearings=1.0,
        spreads=0.0,
        in_arrears=False,
        redemption=100.0,
        issue_date=issue,
    )


def test_floating_rate_bond_structure(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _build_floater()
    assert len(bond.cashflows()) == ref["n_cashflows"]


def test_floating_rate_bond_settle(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _build_floater()
    assert bond.settlement_date().serial_number() == ref["settle_serial"]


def test_floating_rate_bond_notional(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _build_floater()
    assert bond.notional(bond.settlement_date()) == ref["notional"]


def test_floating_rate_bond_npv(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _build_floater()
    disc_curve = FlatForward.from_rate(
        Date.from_ymd(1, Month.December, 2024),
        0.04,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
    )
    bond.set_pricing_engine(DiscountingBondEngine(disc_curve))
    # IborCoupon needs a pricer
    set_coupon_pricer(bond.cashflows(), BlackIborCouponPricer())
    tolerance.tight(bond.npv(), ref["npv"])


def test_floating_rate_bond_clean_price(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _build_floater()
    disc_curve = FlatForward.from_rate(
        Date.from_ymd(1, Month.December, 2024),
        0.04,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
    )
    bond.set_pricing_engine(DiscountingBondEngine(disc_curve))
    set_coupon_pricer(bond.cashflows(), BlackIborCouponPricer())
    tolerance.tight(bond.clean_price(), ref["clean_price"])


def test_floating_rate_bond_accrued_zero_at_issue(
    pinned_eval_date: Any, ref: dict[str, Any],
) -> None:
    """Settlement = issue date → no accrued."""
    bond = _build_floater()
    disc_curve = FlatForward.from_rate(
        Date.from_ymd(1, Month.December, 2024),
        0.04,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
    )
    bond.set_pricing_engine(DiscountingBondEngine(disc_curve))
    set_coupon_pricer(bond.cashflows(), BlackIborCouponPricer())
    tolerance.tight(bond.accrued_amount(bond.settlement_date()), ref["accrued_amount"])
