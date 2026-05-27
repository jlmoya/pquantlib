"""Tests for ZeroCouponBond + DiscountingBondEngine."""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.instruments.bonds.zero_coupon_bond import ZeroCouponBond
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.bond.discounting_bond_engine import DiscountingBondEngine
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l3b")["zero_coupon_bond"]


@pytest.fixture
def pinned_eval_date() -> Any:
    settings = ObservableSettings()
    saved = settings.evaluation_date
    settings.evaluation_date = Date.from_ymd(15, Month.January, 2025)
    yield None
    settings.evaluation_date = saved


def _zcb() -> ZeroCouponBond:
    return ZeroCouponBond(
        settlement_days=2,
        calendar=TARGET(),
        face_amount=100.0,
        maturity_date=Date.from_ymd(15, Month.January, 2030),
        payment_convention=BusinessDayConvention.Following,
        redemption=100.0,
        issue_date=Date.from_ymd(15, Month.January, 2025),
    )


def test_zcb_structure() -> None:
    bond = _zcb()
    # Single redemption only — no coupons.
    assert len(bond.cashflows()) == 1
    assert len(bond.redemptions()) == 1


def test_zcb_settle_serial(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _zcb()
    assert bond.settlement_date().serial_number() == ref["settle_serial"]


def test_zcb_redemption_date_adjusted(
    pinned_eval_date: Any, ref: dict[str, Any],
) -> None:
    bond = _zcb()
    # Single redemption — its date is the adjusted maturity.
    redemption_serial = bond.redemption().date().serial_number()
    assert redemption_serial == ref["redemption_serial"]


def test_zcb_clean_price(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _zcb()
    curve = FlatForward.from_rate(
        Date.from_ymd(15, Month.January, 2025),
        0.05,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
    )
    bond.set_pricing_engine(DiscountingBondEngine(curve))
    tolerance.tight(bond.clean_price(), ref["clean_price"])
    # Zero-coupon bond → clean == dirty (no accrued).
    tolerance.tight(bond.dirty_price(), ref["dirty_price"])


def test_zcb_npv(pinned_eval_date: Any, ref: dict[str, Any]) -> None:
    bond = _zcb()
    curve = FlatForward.from_rate(
        Date.from_ymd(15, Month.January, 2025),
        0.05,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
    )
    bond.set_pricing_engine(DiscountingBondEngine(curve))
    tolerance.tight(bond.npv(), ref["npv"])


def test_zcb_settlement_value_equals_dirty_price_for_par_notional(
    pinned_eval_date: Any, ref: dict[str, Any],
) -> None:
    """For face=100, settlement_value = dirty_price (per the C++ formula
    ``dirty_price = settlement_value * 100 / notional``).

    Note ``expected_settlement_value`` in the reference JSON is
    ``redemption * df(maturity)`` -- the *raw* discounted redemption,
    discounted to the curve's reference date. It equals the engine's
    ``results.value`` (NPV at valuation_date) but NOT ``settlement_value``
    (which is the NPV re-discounted to settlement_date).
    """
    bond = _zcb()
    curve = FlatForward.from_rate(
        Date.from_ymd(15, Month.January, 2025),
        0.05,
        Actual365Fixed(),
        Compounding.Compounded,
        Frequency.Annual,
    )
    bond.set_pricing_engine(DiscountingBondEngine(curve))
    tolerance.tight(bond.settlement_value(), ref["dirty_price"])
    # And value (NPV) equals df * 100 (per the C++ formula above).
    tolerance.tight(bond.npv(), ref["expected_settlement_value"])
