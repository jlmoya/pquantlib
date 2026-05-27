"""Tests for FixedRateBond + DiscountingBondEngine integration.

Cross-validated against the C++ ``cluster_l3b`` probe values in
``migration-harness/references/cluster/l3b.json``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.instruments.bond import BondPrice, BondPriceType
from pquantlib.instruments.bonds.fixed_rate_bond import FixedRateBond
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
    return reference_reader.load("cluster/l3b")["fixed_rate_bond"]


@pytest.fixture(scope="module")
def ref_accrued() -> dict[str, Any]:
    return reference_reader.load("cluster/l3b")["accrued_mid_period"]


def _build_bond() -> FixedRateBond:
    """5y semiannual 5% bond, issued/matures Jan 15, 2025/2030."""
    issue = Date.from_ymd(15, Month.January, 2025)
    maturity = Date.from_ymd(15, Month.January, 2030)
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
    return FixedRateBond(
        settlement_days=2,
        face_amount=100.0,
        schedule=schedule,
        coupons=[0.05],
        accrual_day_counter=Thirty360(Convention.BondBasis),
        payment_convention=BusinessDayConvention.Following,
        redemption=100.0,
        issue_date=issue,
    )


@pytest.fixture
def pinned_eval_date() -> Any:
    """Pin evaluation date to match the C++ probe, then restore."""
    settings = ObservableSettings()
    saved = settings.evaluation_date
    settings.evaluation_date = Date.from_ymd(15, Month.January, 2025)
    yield None
    settings.evaluation_date = saved


# --- construction --------------------------------------------------------


def test_fixed_rate_bond_constructs(pinned_eval_date: Any) -> None:
    bond = _build_bond()
    # 5y * 2 = 10 coupons + 1 redemption.
    assert len(bond.cashflows()) == 11
    assert bond.frequency() == Frequency.Semiannual


def test_fixed_rate_bond_serial_dates(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _build_bond()
    settle = bond.settlement_date()
    assert settle.serial_number() == ref["settle_serial"]
    assert bond.issue_date().serial_number() == ref["issue_serial"]
    assert bond.maturity_date().serial_number() == ref["maturity_serial"]


def test_fixed_rate_bond_start_and_maturity_date(
    pinned_eval_date: Any, ref: dict[str, Any],
) -> None:
    bond = _build_bond()
    assert bond.start_date().serial_number() == ref["start_date_serial"]
    assert bond.maturity_date().serial_number() == ref["maturity_date_serial"]


def test_fixed_rate_bond_is_tradable(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _build_bond()
    assert bool(ref["is_tradable"]) == bond.is_tradable()


def test_fixed_rate_bond_is_expired(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _build_bond()
    assert bool(ref["is_expired"]) == bond.is_expired()


# --- engine-driven prices ------------------------------------------------


def test_fixed_rate_bond_clean_price(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _build_bond()
    curve = FlatForward.from_rate(
        Date.from_ymd(15, Month.January, 2025),
        0.05,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
    )
    bond.set_pricing_engine(DiscountingBondEngine(curve))
    tolerance.tight(bond.clean_price(), ref["clean_price"])


def test_fixed_rate_bond_dirty_price(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _build_bond()
    curve = FlatForward.from_rate(
        Date.from_ymd(15, Month.January, 2025),
        0.05,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
    )
    bond.set_pricing_engine(DiscountingBondEngine(curve))
    tolerance.tight(bond.dirty_price(), ref["dirty_price"])


def test_fixed_rate_bond_settlement_value(
    pinned_eval_date: Any, ref: dict[str, Any],
) -> None:
    bond = _build_bond()
    curve = FlatForward.from_rate(
        Date.from_ymd(15, Month.January, 2025),
        0.05,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
    )
    bond.set_pricing_engine(DiscountingBondEngine(curve))
    tolerance.tight(bond.settlement_value(), ref["settlement_value"])


def test_fixed_rate_bond_npv(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _build_bond()
    curve = FlatForward.from_rate(
        Date.from_ymd(15, Month.January, 2025),
        0.05,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
    )
    bond.set_pricing_engine(DiscountingBondEngine(curve))
    tolerance.tight(bond.npv(), ref["npv"])


def test_fixed_rate_bond_accrued_at_settle(
    pinned_eval_date: Any, ref: dict[str, Any],
) -> None:
    bond = _build_bond()
    accrued = bond.accrued_amount(bond.settlement_date())
    tolerance.tight(accrued, ref["accrued_amount"])


def test_fixed_rate_bond_notional_at_settle(
    pinned_eval_date: Any, ref: dict[str, Any],
) -> None:
    bond = _build_bond()
    assert bond.notional(bond.settlement_date()) == ref["notional"]


# --- yield round-trip ---------------------------------------------------


def test_fixed_rate_bond_yield_solver(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    """Iterative yield solver — LOOSE tier since Brent has wider tol."""
    bond = _build_bond()
    curve = FlatForward.from_rate(
        Date.from_ymd(15, Month.January, 2025),
        0.05,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
    )
    bond.set_pricing_engine(DiscountingBondEngine(curve))
    y = bond.yield_rate(
        Thirty360(Convention.BondBasis),
        Compounding.Compounded,
        Frequency.Semiannual,
    )
    tolerance.loose(y, ref["yield"])


def test_fixed_rate_bond_price_from_yield(
    pinned_eval_date: Any, ref: dict[str, Any],
) -> None:
    """clean_price(yield) round-trip — LOOSE tier (yield itself loose)."""
    bond = _build_bond()
    curve = FlatForward.from_rate(
        Date.from_ymd(15, Month.January, 2025),
        0.05,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
    )
    bond.set_pricing_engine(DiscountingBondEngine(curve))
    y = ref["yield"]
    clean = bond.clean_price_from_yield(
        y,
        Thirty360(Convention.BondBasis),
        Compounding.Compounded,
        Frequency.Semiannual,
    )
    tolerance.loose(clean, ref["clean_price_from_yield"])


def test_fixed_rate_bond_yield_from_price() -> None:
    """yield_from_price(BondPrice) inverts clean_price_from_yield."""
    settings = ObservableSettings()
    saved = settings.evaluation_date
    try:
        settings.evaluation_date = Date.from_ymd(15, Month.January, 2025)
        bond = _build_bond()
        dc = Thirty360(Convention.BondBasis)
        # Forward direction: yield → clean price
        target_yield = 0.06
        clean = bond.clean_price_from_yield(
            target_yield, dc, Compounding.Compounded, Frequency.Semiannual
        )
        # Inverse: solve for yield from clean price
        price = BondPrice(clean, BondPriceType.Clean)
        recovered = bond.yield_from_price(
            price, dc, Compounding.Compounded, Frequency.Semiannual,
        )
        tolerance.loose(recovered, target_yield)
    finally:
        settings.evaluation_date = saved


# --- accrued + next/previous coupon at mid-period ------------------------


def test_accrued_amount_mid_period(pinned_eval_date: Any, ref_accrued: dict[str, Any]) -> None:
    """Settlement mid-first-period (Apr 15, 2025) → accrued ≈ 1.25."""
    bond = _build_bond()
    mid = Date.from_ymd(15, Month.April, 2025)
    accrued = bond.accrued_amount(mid)
    tolerance.tight(accrued, ref_accrued["accrued_amount"])


def test_next_cash_flow_date(pinned_eval_date: Any, ref_accrued: dict[str, Any]) -> None:
    bond = _build_bond()
    mid = Date.from_ymd(15, Month.April, 2025)
    nxt = bond.next_cash_flow_date(mid)
    assert nxt.serial_number() == ref_accrued["next_cf_serial"]


def test_previous_cash_flow_date(
    pinned_eval_date: Any, ref_accrued: dict[str, Any],
) -> None:
    """No previous flow yet (Apr 15, 2025 is still in coupon 0)."""
    bond = _build_bond()
    mid = Date.from_ymd(15, Month.April, 2025)
    prev = bond.previous_cash_flow_date(mid)
    # Null date serial-number is 0 in C++; pquantlib mirrors.
    assert prev.serial_number() == ref_accrued["prev_cf_serial"]


def test_next_coupon_rate(pinned_eval_date: Any, ref_accrued: dict[str, Any]) -> None:
    bond = _build_bond()
    mid = Date.from_ymd(15, Month.April, 2025)
    rate = bond.next_coupon_rate(mid)
    tolerance.tight(rate, ref_accrued["next_coupon_rate"])
