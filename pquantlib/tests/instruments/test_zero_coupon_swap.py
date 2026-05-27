"""Tests for ZeroCouponSwap.

Reference values: ``migration-harness/references/cluster/l3c.json`` →
``zero_coupon_swap_5y``.

ZeroCouponSwap uses an internal ``_CompoundedIborCashFlow`` placeholder
that walks the index's sub-period schedule rather than building a full
MultipleResetsCoupon — see ``zero_coupon_swap.py`` for the carve-out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.zero_coupon_swap import ZeroCouponSwap
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF_PATH = Path(__file__).resolve().parents[3] / "migration-harness/references/cluster/l3c.json"


@pytest.fixture(scope="module")
def cluster_refs() -> dict[str, dict[str, float]]:
    return json.loads(_REF_PATH.read_text())


def _build_5y_zcs() -> tuple[ZeroCouponSwap, YieldTermStructureProtocol]:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    idx = Euribor(Period(3, TimeUnit.Months), curve)
    cal = TARGET()
    start = cal.advance(eval_date, 2, TimeUnit.Days)
    maturity = start + Period(5, TimeUnit.Years)
    zcs = ZeroCouponSwap(
        SwapType.Payer, 1_000_000.0, start, maturity,
        fixed_payment_or_rate=0.05, ibor_index=idx,
        payment_calendar=TARGET(),
        payment_convention=BusinessDayConvention.ModifiedFollowing,
        payment_delay=0,
        fixed_day_counter=Actual360(),
    )
    zcs.set_pricing_engine(DiscountingSwapEngine(curve))
    return zcs, curve


def test_zcs_fixed_payment(cluster_refs: dict[str, dict[str, float]]) -> None:
    zcs, _ = _build_5y_zcs()
    expected = cluster_refs["zero_coupon_swap_5y"]
    tolerance.loose(zcs.fixed_payment(), expected["fixed_payment"])


def test_zcs_fixed_leg_npv(cluster_refs: dict[str, dict[str, float]]) -> None:
    zcs, _ = _build_5y_zcs()
    expected = cluster_refs["zero_coupon_swap_5y"]
    tolerance.loose(zcs.fixed_leg_npv(), expected["fixed_leg_npv"])


def test_zcs_floating_leg_npv(cluster_refs: dict[str, dict[str, float]]) -> None:
    zcs, _ = _build_5y_zcs()
    expected = cluster_refs["zero_coupon_swap_5y"]
    tolerance.loose(zcs.floating_leg_npv(), expected["floating_leg_npv"])


def test_zcs_npv(cluster_refs: dict[str, dict[str, float]]) -> None:
    zcs, _ = _build_5y_zcs()
    expected = cluster_refs["zero_coupon_swap_5y"]
    tolerance.loose(zcs.npv(), expected["npv"])


def test_zcs_fair_fixed_payment(cluster_refs: dict[str, dict[str, float]]) -> None:
    zcs, _ = _build_5y_zcs()
    expected = cluster_refs["zero_coupon_swap_5y"]
    tolerance.loose(zcs.fair_fixed_payment(), expected["fair_fixed_payment"])


def test_zcs_fair_fixed_rate(cluster_refs: dict[str, dict[str, float]]) -> None:
    zcs, _ = _build_5y_zcs()
    expected = cluster_refs["zero_coupon_swap_5y"]
    tolerance.loose(zcs.fair_fixed_rate(Actual360()), expected["fair_fixed_rate"])


def test_zcs_inspectors() -> None:
    zcs, _ = _build_5y_zcs()
    assert zcs.swap_type() == SwapType.Payer
    assert zcs.base_nominal() == 1_000_000.0
    # Fixed leg = single FixedRateCoupon; floating leg = single placeholder.
    assert len(zcs.fixed_leg()) == 1
    assert len(zcs.floating_leg()) == 1
    # Payer convention: fixed paid, float received.
    assert zcs.payer(0)
    assert not zcs.payer(1)


def test_zcs_known_amount_constructor() -> None:
    """Constructor with a known fixed payment amount (not a rate)."""
    eval_date = Date.from_ymd(17, Month.January, 2024)
    curve = cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )
    idx = Euribor(Period(3, TimeUnit.Months), curve)
    cal = TARGET()
    start = cal.advance(eval_date, 2, TimeUnit.Days)
    maturity = start + Period(5, TimeUnit.Years)
    zcs = ZeroCouponSwap(
        SwapType.Payer, 1_000_000.0, start, maturity,
        fixed_payment_or_rate=280_960.36678731552, ibor_index=idx,
        payment_calendar=TARGET(),
    )
    # No fixed_day_counter -> SimpleCashFlow with the known amount.
    assert zcs.fixed_payment() == 280_960.36678731552
