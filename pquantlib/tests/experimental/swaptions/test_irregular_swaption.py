"""Tests for the irregular-swap / irregular-swaption family (W8-A cluster b).

# C++ parity:
#   ql/experimental/swaptions/irregularswap.hpp
#   ql/experimental/swaptions/irregularswaption.hpp
#   ql/experimental/swaptions/haganirregularswaptionengine.hpp

Cross-validates IrregularSwap NPV (step-down notional) and the
HaganIrregularSwaptionEngine NPV against
``migration-harness/references/cluster/w8a.json``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.cashflows.coupon_pricer import (
    BlackIborCouponPricer,
    set_coupon_pricer,
)
from pquantlib.cashflows.fixed_rate_leg import fixed_rate_leg
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention as T360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.irregular_swap import IrregularSwap
from pquantlib.instruments.irregular_swaption import (
    IrregularSettlement,
    IrregularSwaption,
)
from pquantlib.instruments.swap import SwapType
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.pricingengines.swaption.hagan_irregular_swaption_engine import (
    HaganIrregularSwaptionEngine,
)
from pquantlib.termstructures.volatility.swaption.swaption_constant_vol import (
    SwaptionConstantVolatility,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/w8a")


def _d(day: int, month: Month, year: int) -> Date:
    return Date.from_ymd(day, month, year)


_TODAY = _d(15, Month.January, 2024)


def _curve(rate: float = 0.03) -> FlatForward:
    return FlatForward.from_rate(_TODAY, rate, Actual365Fixed())


def _build_irregular_swap(
    curve: FlatForward, start: Date, maturity: Date
) -> IrregularSwap:
    """Reproduce the probe's step-down IrregularSwap."""
    cal = TARGET()
    euribor6m = Euribor.six_months(curve)

    fixed_sched = Schedule.from_rule(
        start, maturity, Period(1, TimeUnit.Years), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Forward, False,
    )
    float_sched = Schedule.from_rule(
        start, maturity, Period(6, TimeUnit.Months), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Forward, False,
    )

    fixed_leg = fixed_rate_leg(
        fixed_sched,
        nominals=[1.0e6, 0.7e6, 0.4e6],
        rates=[0.035],
        day_counter=Thirty360(T360Convention.BondBasis),
        payment_adjustment=BusinessDayConvention.ModifiedFollowing,
    )
    float_leg = ibor_leg(
        float_sched,
        euribor6m,
        nominals=[1.0e6, 1.0e6, 0.7e6, 0.7e6, 0.4e6, 0.4e6],
        payment_day_counter=Actual360(),
        payment_adjustment=BusinessDayConvention.ModifiedFollowing,
    )
    # C++ IborLeg attaches a BlackIborCouponPricer; mirror it so amount() works.
    set_coupon_pricer(float_leg, BlackIborCouponPricer())
    return IrregularSwap(SwapType.Receiver, fixed_leg, float_leg)


# ---------------------------------------------------------------------------
# IrregularSwap — cross-validated
# ---------------------------------------------------------------------------


def test_irregular_swap_npv(reference_data: dict[str, Any]) -> None:
    """Step-down IrregularSwap NPV + per-leg NPVs vs C++.

    LOOSE: discounted-cashflow swap on a flat curve.
    """
    curve = _curve()
    swap = _build_irregular_swap(
        curve, _d(15, Month.January, 2025), _d(15, Month.January, 2028)
    )
    swap.set_pricing_engine(DiscountingSwapEngine(curve))

    loose(swap.npv(), reference_data["irr_swap_npv"])
    loose(swap.fixed_leg_npv(), reference_data["irr_swap_fixed_npv"])
    loose(swap.floating_leg_npv(), reference_data["irr_swap_float_npv"])


def test_irregular_swap_type_and_legs() -> None:
    curve = _curve()
    swap = _build_irregular_swap(
        curve, _d(15, Month.January, 2025), _d(15, Month.January, 2028)
    )
    assert swap.type() == SwapType.Receiver
    assert len(swap.fixed_leg()) == 3
    assert len(swap.floating_leg()) == 6
    # Receiver: fixed received (+1), floating paid (-1).
    assert swap.payer(0) is False
    assert swap.payer(1) is True


def test_irregular_swap_payer_signs() -> None:
    curve = _curve()
    cal = TARGET()
    euribor6m = Euribor.six_months(curve)
    start = _d(15, Month.January, 2025)
    maturity = _d(15, Month.January, 2027)
    fixed_sched = Schedule.from_rule(
        start, maturity, Period(1, TimeUnit.Years), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing, DateGeneration.Forward, False,
    )
    float_sched = Schedule.from_rule(
        start, maturity, Period(6, TimeUnit.Months), cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing, DateGeneration.Forward, False,
    )
    fixed_leg = fixed_rate_leg(
        fixed_sched, nominals=[1.0e6], rates=[0.035],
        day_counter=Thirty360(T360Convention.BondBasis),
    )
    float_leg = ibor_leg(float_sched, euribor6m, nominals=[1.0e6])
    payer = IrregularSwap(SwapType.Payer, fixed_leg, float_leg)
    # Payer: fixed paid (-1), floating received (+1).
    assert payer.payer(0) is True
    assert payer.payer(1) is False


# ---------------------------------------------------------------------------
# IrregularSwaption + HaganIrregularSwaptionEngine — cross-validated
# ---------------------------------------------------------------------------


def _hagan_setup() -> tuple[FlatForward, IrregularSwaption, HaganIrregularSwaptionEngine]:
    curve = _curve()
    cal = TARGET()
    start = _d(15, Month.January, 2025)
    maturity = _d(15, Month.January, 2028)
    swap = _build_irregular_swap(curve, start, maturity)

    vol = SwaptionConstantVolatility(
        reference_date=_TODAY,
        calendar=cal,
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=0.0080,
        day_counter=Actual365Fixed(),
        volatility_type=VolatilityType.Normal,
    )

    exercise = EuropeanExercise(start)
    swaption = IrregularSwaption(swap, exercise)
    engine = HaganIrregularSwaptionEngine(vol, curve)
    swaption.set_pricing_engine(engine)
    return curve, swaption, engine


def test_hagan_swaption_npv(reference_data: dict[str, Any]) -> None:
    """Hagan irregular-swaption NPV vs C++.

    LOOSE: Hagan linear-TSR super-replication basket (lstsq solve + sum of
    Bachelier vanilla-swaption prices).
    """
    _curve_obj, swaption, _engine = _hagan_setup()
    loose(swaption.npv(), reference_data["hagan_swaption_npv"])


def test_hagan_swaption_positive() -> None:
    _curve_obj, swaption, _engine = _hagan_setup()
    assert swaption.npv() > 0.0


def test_hagan_rejects_non_european() -> None:
    """The engine requires a European exercise."""
    curve = _curve()
    cal = TARGET()
    start = _d(15, Month.January, 2025)
    maturity = _d(15, Month.January, 2028)
    swap = _build_irregular_swap(curve, start, maturity)
    vol = SwaptionConstantVolatility(
        reference_date=_TODAY, calendar=cal,
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=0.0080, day_counter=Actual365Fixed(),
        volatility_type=VolatilityType.Normal,
    )
    exercise = AmericanExercise(start, maturity)
    swaption = IrregularSwaption(swap, exercise)
    swaption.set_pricing_engine(HaganIrregularSwaptionEngine(vol, curve))
    with pytest.raises(LibraryException, match="european"):
        swaption.npv()


# ---------------------------------------------------------------------------
# IrregularSwaption — structural
# ---------------------------------------------------------------------------


def test_irregular_swaption_inspectors() -> None:
    curve = _curve()
    swap = _build_irregular_swap(
        curve, _d(15, Month.January, 2025), _d(15, Month.January, 2028)
    )
    exercise = EuropeanExercise(_d(15, Month.January, 2025))
    swaption = IrregularSwaption(swap, exercise, IrregularSettlement.Type.Cash)
    assert swaption.underlying_swap() is swap
    assert swaption.settlement_type() == IrregularSettlement.Type.Cash
    assert swaption.type() == SwapType.Receiver
    assert swaption.is_expired() is False


def test_irregular_settlement_to_string() -> None:
    assert IrregularSettlement.to_string(IrregularSettlement.Type.Physical) == "Delivery"
    assert IrregularSettlement.to_string(IrregularSettlement.Type.Cash) == "Cash"
