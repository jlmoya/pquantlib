"""Tests for the basis-models cluster (W8-A cluster c).

# C++ parity:
#   ql/experimental/basismodels/swaptioncfs.hpp
#   ql/experimental/basismodels/tenorswaptionvts.hpp
#   ql/experimental/basismodels/tenoroptionletvts.hpp

Cross-validates TenorSwaptionVTS.volatility rescaling and SwaptionCashFlows
weight sums against ``migration-harness/references/cluster/w8a.json``.
TenorOptionletVTS is exercised structurally (see module docstring there for the
smile-section divergence note).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.cashflows.coupon_pricer import BlackIborCouponPricer, set_coupon_pricer
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention as T360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.experimental.basismodels.swaption_cfs import (
    SwaptionCashFlows,
    swaption_cashflows,
)
from pquantlib.experimental.basismodels.tenor_optionlet_vts import (
    TenorOptionletVTS,
    TwoParameterCorrelation,
)
from pquantlib.experimental.basismodels.tenor_swaption_vts import TenorSwaptionVTS
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.instruments.make_vanilla_swap import make_vanilla_swap
from pquantlib.instruments.swaption import Swaption
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.termstructures.volatility.optionlet.constant_optionlet_vol import (
    ConstantOptionletVolatility,
)
from pquantlib.termstructures.volatility.swaption.swaption_constant_vol import (
    SwaptionConstantVolatility,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/w8a")


def _d(day: int, month: Month, year: int) -> Date:
    return Date.from_ymd(day, month, year)


_TODAY = _d(15, Month.January, 2024)


def _curve(rate: float = 0.03) -> FlatForward:
    return FlatForward.from_rate(_TODAY, rate, Actual365Fixed())


# ---------------------------------------------------------------------------
# TenorSwaptionVTS — cross-validated
# ---------------------------------------------------------------------------


def _tenor_swaption_vts(curve: FlatForward) -> TenorSwaptionVTS:
    """Reproduce the probe's TenorSwaptionVTS (base 6M -> target 3M)."""
    cal = TARGET()
    base6m = Euribor.six_months(curve)
    targ3m = Euribor.three_months(curve)
    base_vol = SwaptionConstantVolatility(
        reference_date=_TODAY,
        calendar=cal,
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=0.0090,
        day_counter=Actual365Fixed(),
        volatility_type=VolatilityType.Normal,
    )
    return TenorSwaptionVTS(
        base_vol, curve, base6m, targ3m,
        Period(1, TimeUnit.Years), Period(1, TimeUnit.Years),
        Thirty360(T360Convention.BondBasis), Thirty360(T360Convention.BondBasis),
    )


def test_tenor_swaption_vts_volatility(reference_data: dict[str, Any]) -> None:
    """Rescaled normal vol at (option, swap, strike) vs C++.

    LOOSE: affine-TSR rescaling (3 vanilla-swap fair rates + cash-flow
    decomposition).
    """
    vts = _tenor_swaption_vts(_curve())
    loose(vts.volatility(5.0, 10.0, 0.03), reference_data["tsvts_vol_5x10_atm"])
    loose(vts.volatility(2.0, 5.0, 0.03), reference_data["tsvts_vol_2x5_atm"])
    loose(vts.volatility(5.0, 10.0, 0.04), reference_data["tsvts_vol_5x10_otm"])


def test_tenor_swaption_vts_normal_type() -> None:
    vts = _tenor_swaption_vts(_curve())
    assert vts.volatility_type() == VolatilityType.Normal


def test_tenor_swaption_vts_smile_section() -> None:
    """The smile section round-trips through the rescaled volatility."""
    vts = _tenor_swaption_vts(_curve())
    section = vts.smile_section(5.0, 10.0)
    loose(section.volatility(0.03), vts.volatility(5.0, 10.0, 0.03))
    # atm_level is the final-tenor swap rate (positive on a 3% curve).
    assert section.atm_level() > 0.0


# ---------------------------------------------------------------------------
# SwaptionCashFlows — cross-validated
# ---------------------------------------------------------------------------


def _swaption_cfs(curve: FlatForward) -> SwaptionCashFlows:
    """Reproduce the probe's SwaptionCashFlows (5Y swap)."""
    cal = TARGET()
    euribor6m = Euribor.six_months(curve)
    exercise = _d(15, Month.January, 2026)
    swap = make_vanilla_swap(
        swap_tenor=Period(5, TimeUnit.Years),
        ibor_index=euribor6m,
        fixed_rate=0.03,
        effective_date=cal.advance(exercise, 2, TimeUnit.Days),
        fixed_leg_tenor=Period(1, TimeUnit.Years),
        fixed_leg_day_count=Thirty360(T360Convention.BondBasis),
        discount_curve=curve,
    )
    # the float leg needs a pricer for coupon.rate() in the decomposition.
    set_coupon_pricer(swap.floating_leg(), BlackIborCouponPricer())
    swap.set_pricing_engine(DiscountingSwapEngine(curve))
    swaption = Swaption(swap, EuropeanExercise(exercise))
    return SwaptionCashFlows(swaption, curve)


def test_swaption_cfs_weight_sums(reference_data: dict[str, Any]) -> None:
    """Annuity / float / fixed weight sums vs C++.

    TIGHT: deterministic cash-flow decomposition on a shared flat curve.
    """
    cfs = _swaption_cfs(_curve())
    tight(sum(cfs.annuity_weights()), reference_data["scfs_sum_annuity"])
    tight(sum(cfs.float_weights()), reference_data["scfs_sum_float"])
    tight(sum(cfs.fixed_weights()), reference_data["scfs_sum_fixed"])


def test_swaption_cfs_counts(reference_data: dict[str, Any]) -> None:
    cfs = _swaption_cfs(_curve())
    assert len(cfs.exercise_times()) == int(reference_data["scfs_num_exercise"])
    assert len(cfs.float_times()) == int(reference_data["scfs_num_float"])
    tight(cfs.float_times()[0], reference_data["scfs_first_float_time"])


def test_swaption_cashflows_free_function() -> None:
    """The free function matches the class."""
    curve = _curve()
    cal = TARGET()
    euribor6m = Euribor.six_months(curve)
    exercise = _d(15, Month.January, 2026)
    swap = make_vanilla_swap(
        swap_tenor=Period(5, TimeUnit.Years),
        ibor_index=euribor6m,
        fixed_rate=0.03,
        effective_date=cal.advance(exercise, 2, TimeUnit.Days),
        fixed_leg_tenor=Period(1, TimeUnit.Years),
        fixed_leg_day_count=Thirty360(T360Convention.BondBasis),
        discount_curve=curve,
    )
    set_coupon_pricer(swap.floating_leg(), BlackIborCouponPricer())
    swaption = Swaption(swap, EuropeanExercise(exercise))
    cfs = swaption_cashflows(swaption, curve)
    assert cfs.swaption() is swaption
    assert len(cfs.fixed_times()) == 5


# ---------------------------------------------------------------------------
# TenorOptionletVTS — structural
# ---------------------------------------------------------------------------


def test_tenor_optionlet_vts_volatility_positive() -> None:
    """The tenor-rescaled optionlet vol is finite and positive.

    Uses a NullCalendar base/target so the internally-generated sub-schedule
    never lands a fixing on a weekend (matches the probe setup).
    """
    curve = _curve()
    ncal = NullCalendar()
    base3m = IborIndex(
        "Base3M", Period(3, TimeUnit.Months), 2, EURCurrency(), ncal,
        BusinessDayConvention.ModifiedFollowing, False, Actual360(), curve,
    )
    targ6m = IborIndex(
        "Targ6M", Period(6, TimeUnit.Months), 2, EURCurrency(), ncal,
        BusinessDayConvention.ModifiedFollowing, False, Actual360(), curve,
    )
    base_vol = ConstantOptionletVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=0.0070,
        calendar=ncal,
        day_counter=Actual365Fixed(),
        reference_date=_TODAY,
        volatility_type=VolatilityType.Normal,
    )
    times = np.array([0.0, 30.0])
    rho_inf = LinearInterpolation(times, np.array([0.3, 0.3]))
    beta = LinearInterpolation(times, np.array([0.1, 0.1]))
    corr = TwoParameterCorrelation(rho_inf, beta)

    vts = TenorOptionletVTS(base_vol, base3m, targ6m, corr)
    vol = vts.volatility(5.0, 0.03, True)
    assert vol > 0.0
    assert np.isfinite(vol)
    assert vts.volatility_type() == VolatilityType.Normal


def test_tenor_optionlet_vts_requires_freq_multiple() -> None:
    """baseFreq must be a multiple of targFreq (6M base, 3M target fails)."""
    curve = _curve()
    ncal = NullCalendar()
    base6m = IborIndex(
        "Base6M", Period(6, TimeUnit.Months), 2, EURCurrency(), ncal,
        BusinessDayConvention.ModifiedFollowing, False, Actual360(), curve,
    )
    targ3m = IborIndex(
        "Targ3M", Period(3, TimeUnit.Months), 2, EURCurrency(), ncal,
        BusinessDayConvention.ModifiedFollowing, False, Actual360(), curve,
    )
    base_vol = ConstantOptionletVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=0.0070, calendar=ncal, day_counter=Actual365Fixed(),
        reference_date=_TODAY, volatility_type=VolatilityType.Normal,
    )
    times = np.array([0.0, 30.0])
    corr = TwoParameterCorrelation(
        LinearInterpolation(times, np.array([0.3, 0.3])),
        LinearInterpolation(times, np.array([0.1, 0.1])),
    )
    with pytest.raises(LibraryException, match="multiple of target"):
        TenorOptionletVTS(base_vol, base6m, targ3m, corr)
