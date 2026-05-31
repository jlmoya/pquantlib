"""Callable bonds (W8-B batch a) — cross-validation vs C++ probe.

Probe source: migration-harness/cpp/probes/cluster_w8b/probe.cpp
Reference:    migration-harness/references/cluster/w8b.json

A 5y annual 6% bond, callable/puttable at par in 3y, priced under a flat
5.5% continuous curve with:
  * BlackCallableFixedRateBondEngine @ 20% fwd-yield vol, and
  * TreeCallableFixedRateBondEngine under HullWhite(a=0.03, sigma=0.012)
    @ 40 timesteps.
Also exercises the OAS / cleanPriceOAS / effectiveDuration round-trip and
the CallableBondConstantVolatility inspectors.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.instruments.bond import BondPrice, BondPriceType
from pquantlib.instruments.callability import Callability, CallabilityType
from pquantlib.instruments.callable_bond import (
    CallableFixedRateBond,
    CallableZeroCouponBond,
)
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.bond.black_callable_bond_engine import (
    BlackCallableFixedRateBondEngine,
)
from pquantlib.pricingengines.bond.tree_callable_bond_engine import (
    TreeCallableFixedRateBondEngine,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.callable_bond_constant_vol import (
    CallableBondConstantVolatility,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w8b")


_TODAY = Date.from_ymd(15, Month.February, 2007)
_ISSUE = Date.from_ymd(15, Month.February, 2007)
_MATURITY = Date.from_ymd(15, Month.February, 2012)
_CALL_DATE = Date.from_ymd(15, Month.February, 2010)


@pytest.fixture(autouse=True)
def _set_eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    ObservableSettings().evaluation_date = _TODAY


def _curve() -> FlatForward:
    return FlatForward.from_rate(_TODAY, 0.055, Actual365Fixed(), Compounding.Continuous, Frequency.Annual)


def _schedule() -> Schedule:
    return Schedule.from_rule(
        _ISSUE,
        _MATURITY,
        Period(1, TimeUnit.Years),
        NullCalendar(),
        BusinessDayConvention.Unadjusted,
        BusinessDayConvention.Unadjusted,
        DateGeneration.Backward,
        False,
    )


def _bond(call_type: CallabilityType = CallabilityType.Call) -> CallableFixedRateBond:
    call = Callability(
        BondPrice(100.0, BondPriceType.Clean), call_type, _CALL_DATE
    )
    return CallableFixedRateBond(
        settlement_days=0,
        face_amount=100.0,
        schedule=_schedule(),
        coupons=[0.06],
        accrual_day_counter=Thirty360(Thirty360Convention.BondBasis),
        payment_convention=BusinessDayConvention.Following,
        redemption=100.0,
        issue_date=_ISSUE,
        put_call_schedule=[call],
    )


# ----------------------------------------------------------------------
# Black engine
# ----------------------------------------------------------------------


def test_black_callable_npv(cpp_ref: dict[str, Any]) -> None:
    """BlackCallableFixedRateBondEngine NPV @ 20% fwd-yield vol vs C++."""
    bond = _bond()
    vol = SimpleQuote(0.20)
    bond.set_pricing_engine(BlackCallableFixedRateBondEngine(vol, _curve()))

    tolerance.loose(bond.npv(), float(cpp_ref["black_npv"]))
    tolerance.loose(bond.settlement_value(), float(cpp_ref["black_settlement_value"]))
    tolerance.loose(bond.clean_price(), float(cpp_ref["black_clean_price"]))


def test_black_callable_via_vol_structure(cpp_ref: dict[str, Any]) -> None:
    """The vol-structure ctor overload matches the Quote ctor."""
    bond = _bond()
    vol_ts = CallableBondConstantVolatility(
        0.20, Actual365Fixed(), settlement_days=0, calendar=NullCalendar()
    )
    bond.set_pricing_engine(BlackCallableFixedRateBondEngine(vol_ts, _curve()))
    tolerance.loose(bond.npv(), float(cpp_ref["black_npv"]))


# ----------------------------------------------------------------------
# Tree engine
# ----------------------------------------------------------------------


# Tree-discretisation gap: the Python and C++ HullWhite short-rate trees
# build the same lattice topology at N=40, but state-price accumulation /
# floating-point rounding differs at the ~1e-6 absolute level (= ~1e-8
# relative on a ~100 price). This matches the documented tolerance the
# existing TreeSwaptionEngine cross-validation uses. The agreement to 6
# significant figures proves the callable-bond rollback logic is correct.
_TREE_ABS = 1e-5
_TREE_REL = 1e-6
_TREE_REASON = "HW tree state-price rounding @ N=40 vs C++ tree (6 sig-fig agreement)"


def test_tree_callable_npv(cpp_ref: dict[str, Any]) -> None:
    """TreeCallableFixedRateBondEngine NPV under HullWhite vs C++ probe."""
    bond = _bond()
    hw = HullWhite(_curve(), a=float(cpp_ref["tree_hw_a"]), sigma=float(cpp_ref["tree_hw_sigma"]))
    bond.set_pricing_engine(
        TreeCallableFixedRateBondEngine(hw, int(cpp_ref["tree_steps"]), term_structure=_curve())
    )
    tolerance.custom(bond.npv(), float(cpp_ref["tree_npv"]), abs_tol=_TREE_ABS, rel_tol=_TREE_REL, reason=_TREE_REASON)
    tolerance.custom(
        bond.settlement_value(), float(cpp_ref["tree_settlement_value"]),
        abs_tol=_TREE_ABS, rel_tol=_TREE_REL, reason=_TREE_REASON,
    )
    tolerance.custom(
        bond.clean_price(), float(cpp_ref["tree_clean_price"]),
        abs_tol=_TREE_ABS, rel_tol=_TREE_REL, reason=_TREE_REASON,
    )


def test_tree_puttable_npv(cpp_ref: dict[str, Any]) -> None:
    """Puttable variant floors the value above the plain bond — vs C++."""
    bond = _bond(CallabilityType.Put)
    hw = HullWhite(_curve(), a=float(cpp_ref["tree_hw_a"]), sigma=float(cpp_ref["tree_hw_sigma"]))
    bond.set_pricing_engine(
        TreeCallableFixedRateBondEngine(hw, int(cpp_ref["tree_steps"]), term_structure=_curve())
    )
    tolerance.custom(bond.npv(), float(cpp_ref["tree_put_npv"]), abs_tol=_TREE_ABS, rel_tol=_TREE_REL, reason=_TREE_REASON)


def test_tree_oas_and_clean_price_roundtrip(cpp_ref: dict[str, Any]) -> None:
    """OAS at the model price ≈ 0; cleanPriceOAS(0) recovers the model price."""
    bond = _bond()
    curve = _curve()
    hw = HullWhite(curve, a=float(cpp_ref["tree_hw_a"]), sigma=float(cpp_ref["tree_hw_sigma"]))
    bond.set_pricing_engine(
        TreeCallableFixedRateBondEngine(hw, int(cpp_ref["tree_steps"]), term_structure=curve)
    )
    model_clean = bond.clean_price()

    oas = bond.oas(model_clean, curve, Actual365Fixed(), Compounding.Continuous, Frequency.NoFrequency)
    tolerance.custom(
        oas,
        float(cpp_ref["tree_oas_at_model_price"]),
        abs_tol=1e-7,
        rel_tol=1e-7,
        reason="OAS at model price is a Brent root near 0 — abs tol vs C++ Brent",
    )

    clean_at_zero = bond.clean_price_oas(
        0.0, curve, Actual365Fixed(), Compounding.Continuous, Frequency.NoFrequency
    )
    tolerance.custom(
        clean_at_zero, float(cpp_ref["tree_clean_at_zero_oas"]),
        abs_tol=_TREE_ABS, rel_tol=_TREE_REL, reason=_TREE_REASON,
    )


def test_tree_effective_duration(cpp_ref: dict[str, Any]) -> None:
    """Effective duration (bump-and-revalue) vs C++ probe."""
    bond = _bond()
    curve = _curve()
    hw = HullWhite(curve, a=float(cpp_ref["tree_hw_a"]), sigma=float(cpp_ref["tree_hw_sigma"]))
    bond.set_pricing_engine(
        TreeCallableFixedRateBondEngine(hw, int(cpp_ref["tree_steps"]), term_structure=curve)
    )
    eff_dur = bond.effective_duration(
        0.0, curve, Actual365Fixed(), Compounding.Continuous, Frequency.NoFrequency
    )
    tolerance.loose(eff_dur, float(cpp_ref["tree_effective_duration"]))


# ----------------------------------------------------------------------
# Constant vol inspectors
# ----------------------------------------------------------------------


def test_constant_vol_inspectors(cpp_ref: dict[str, Any]) -> None:
    cv = CallableBondConstantVolatility(0.20, Actual365Fixed(), reference_date=_TODAY)
    tolerance.tight(cv.volatility(1.0, 4.0, 100.0), float(cpp_ref["constvol_vol"]))
    tolerance.tight(cv.black_variance(2.0, 3.0, 100.0), float(cpp_ref["constvol_black_variance"]))


def test_callable_zero_coupon_constructs() -> None:
    """CallableZeroCouponBond builds + sets a single redemption."""
    call = Callability(BondPrice(100.0, BondPriceType.Clean), CallabilityType.Call, _CALL_DATE)
    zcb = CallableZeroCouponBond(
        settlement_days=0,
        face_amount=100.0,
        calendar=NullCalendar(),
        maturity_date=_MATURITY,
        day_counter=Actual365Fixed(),
        redemption=100.0,
        issue_date=_ISSUE,
        put_call_schedule=[call],
    )
    assert zcb.frequency() == Frequency.Once
    assert len(zcb.redemptions()) == 1
