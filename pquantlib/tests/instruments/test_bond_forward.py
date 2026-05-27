"""Tests for BondForward (instrument-only data carrier).

The bond's pricing engine is wired in so dirty_price reflects the
discount-curve NPV, then BondForward computes forward_price /
clean_forward_price / spot_income via the L3-B formulas.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.bond_forward import BondForward, BondForwardPosition
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
    return reference_reader.load("cluster/l3b")["bond_forward"]


@pytest.fixture
def pinned_eval_date() -> Any:
    settings = ObservableSettings()
    saved = settings.evaluation_date
    settings.evaluation_date = Date.from_ymd(15, Month.January, 2025)
    yield None
    settings.evaluation_date = saved


def _build() -> tuple[BondForward, FixedRateBond, FlatForward]:
    issue = Date.from_ymd(15, Month.January, 2025)
    bond_maturity = Date.from_ymd(15, Month.January, 2030)
    schedule = Schedule.from_rule(
        effective_date=issue,
        termination_date=bond_maturity,
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
        Compounding.Continuous,
        Frequency.Annual,
    )
    bond.set_pricing_engine(DiscountingBondEngine(curve))
    bf = BondForward(
        value_date=Date.from_ymd(17, Month.January, 2025),
        maturity_date=Date.from_ymd(15, Month.April, 2025),
        position_type=BondForwardPosition.Long,
        strike=100.0,
        settlement_days=2,
        day_counter=Actual365Fixed(),
        calendar=TARGET(),
        business_day_convention=BusinessDayConvention.Following,
        bond=bond,
        discount_curve=curve,
        income_discount_curve=curve,
    )
    return bf, bond, curve


def test_negative_strike_raises() -> None:
    """C++ parity — ForwardTypePayoff requires strike >= 0."""
    issue = Date.from_ymd(15, Month.January, 2025)
    bond_maturity = Date.from_ymd(15, Month.January, 2030)
    schedule = Schedule.from_rule(
        effective_date=issue,
        termination_date=bond_maturity,
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
        issue, 0.05, Actual365Fixed(), Compounding.Continuous, Frequency.Annual,
    )
    with pytest.raises(LibraryException, match="negative strike"):
        BondForward(
            value_date=Date.from_ymd(17, Month.January, 2025),
            maturity_date=Date.from_ymd(15, Month.April, 2025),
            position_type=BondForwardPosition.Long,
            strike=-1.0,
            settlement_days=2,
            day_counter=Actual365Fixed(),
            calendar=TARGET(),
            business_day_convention=BusinessDayConvention.Following,
            bond=bond,
            discount_curve=curve,
        )


def test_bond_forward_spot_value(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bf, _bond, _curve = _build()
    tolerance.tight(bf.spot_value(), ref["spot_value"])


def test_bond_forward_spot_income_zero_in_window(
    pinned_eval_date: Any, ref: dict[str, Any],
) -> None:
    """No bond coupon falls in (settlement, delivery] → spot_income = 0."""
    bf, _bond, curve = _build()
    tolerance.tight(bf.spot_income(curve), ref["spot_income"])


def test_bond_forward_forward_price(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bf, _bond, _curve = _build()
    tolerance.tight(bf.forward_price(), ref["forward_price"])


def test_bond_forward_clean_forward_price(
    pinned_eval_date: Any, ref: dict[str, Any],
) -> None:
    bf, _bond, _curve = _build()
    tolerance.tight(bf.clean_forward_price(), ref["clean_forward_price"])


def test_bond_forward_strike(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bf, _bond, _curve = _build()
    assert bf.strike() == ref["strike"]


def test_bond_forward_settlement_date_value_date_floor(pinned_eval_date: Any) -> None:
    """settlement_date returns max(eval+2BD, value_date)."""
    bf, _bond, _curve = _build()
    # eval = Jan 15, +2BD on TARGET = Jan 17 = value_date.
    assert bf.settlement_date() == Date.from_ymd(17, Month.January, 2025)


def test_bond_forward_is_expired_false_in_window(pinned_eval_date: Any) -> None:
    """Maturity (Apr 15) is after settlement (Jan 17) → not expired."""
    bf, _bond, _curve = _build()
    assert bf.is_expired() is False


def test_bond_forward_npv_long_short_sign(pinned_eval_date: Any) -> None:
    """NPV of Short == -NPV of Long for the same parameters."""
    bf_long, _bond, _curve = _build()
    long_npv = bf_long.npv()
    # Rebuild with Short position.
    issue = Date.from_ymd(15, Month.January, 2025)
    bond_maturity = Date.from_ymd(15, Month.January, 2030)
    schedule = Schedule.from_rule(
        effective_date=issue,
        termination_date=bond_maturity,
        tenor=Period(6, TimeUnit.Months),
        calendar=TARGET(),
        convention=BusinessDayConvention.Unadjusted,
        termination_date_convention=BusinessDayConvention.Unadjusted,
        rule=DateGeneration.Backward,
        end_of_month=False,
    )
    bond_s = FixedRateBond(
        settlement_days=2, face_amount=100.0, schedule=schedule, coupons=[0.05],
        accrual_day_counter=Thirty360(Convention.BondBasis),
        payment_convention=BusinessDayConvention.Following,
        redemption=100.0, issue_date=issue,
    )
    curve_s = FlatForward.from_rate(
        issue, 0.05, Actual365Fixed(), Compounding.Continuous, Frequency.Annual,
    )
    bond_s.set_pricing_engine(DiscountingBondEngine(curve_s))
    bf_short = BondForward(
        value_date=Date.from_ymd(17, Month.January, 2025),
        maturity_date=Date.from_ymd(15, Month.April, 2025),
        position_type=BondForwardPosition.Short,
        strike=100.0,
        settlement_days=2,
        day_counter=Actual365Fixed(),
        calendar=TARGET(),
        business_day_convention=BusinessDayConvention.Following,
        bond=bond_s,
        discount_curve=curve_s,
        income_discount_curve=curve_s,
    )
    tolerance.tight(bf_short.npv(), -long_npv)
