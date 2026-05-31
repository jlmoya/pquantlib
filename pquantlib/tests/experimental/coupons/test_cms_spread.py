"""Tests for the CMS-spread coupon family (W8-A cluster).

# C++ parity:
#   ql/experimental/coupons/swapspreadindex.hpp
#   ql/experimental/coupons/cmsspreadcoupon.hpp
#   ql/experimental/coupons/proxyibor.hpp
#   ql/experimental/coupons/quantocouponpricer.hpp

Cross-validates SwapSpreadIndex against
``migration-harness/references/cluster/w8a.json``. The CMS-spread coupon shell,
ProxyIbor, and the quanto adjustment are exercised structurally (their full
pricing depends on the deferred CmsCoupon/CmsCouponPricer + cap/floor families
— see docs/carve-outs.md).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.cashflows.cms_spread_coupon import CmsSpreadCoupon
from pquantlib.cashflows.ibor_coupon import IborCoupon
from pquantlib.cashflows.quanto_coupon_pricer import BlackIborQuantoCouponPricer
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention as T360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.ibor.proxy_ibor import ProxyIbor
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.indexes.swap_spread_index import SwapSpreadIndex
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.optionlet.constant_optionlet_vol import (
    ConstantOptionletVolatility,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, loose, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/w8a")


def _d(day: int, month: Month, year: int) -> Date:
    """Local (day, month, year) Date helper (C++ _d(d, m, y) order)."""
    return Date.from_ymd(day, month, year)


_TODAY = _d(15, Month.January, 2024)


def _curve(rate: float = 0.03) -> FlatForward:
    return FlatForward.from_rate(_TODAY, rate, Actual365Fixed())


def _swap_index(tenor_years: int, curve: FlatForward) -> SwapIndex:
    """Reproduce the probe's SwapIndex on Euribor6M."""
    euribor6m = Euribor.six_months(curve)
    return SwapIndex(
        "EuriborSwap",
        Period(tenor_years, TimeUnit.Years),
        2,
        EURCurrency(),
        TARGET(),
        Period(1, TimeUnit.Years),
        BusinessDayConvention.ModifiedFollowing,
        Thirty360(T360Convention.BondBasis),
        euribor6m,
        curve,
    )


# ---------------------------------------------------------------------------
# SwapSpreadIndex — cross-validated
# ---------------------------------------------------------------------------


def test_swap_spread_index_fixing(reference_data: dict[str, Any]) -> None:
    """fixing = g1*fix1 + g2*fix2 vs C++.

    TIGHT: exact algebraic combination of two swap-rate forecast fixings,
    which are themselves discount-factor ratios on a shared flat curve.
    """
    curve = _curve()
    s10 = _swap_index(10, curve)
    s2 = _swap_index(2, curve)
    idx = SwapSpreadIndex("CMS10Y-2Y", s10, s2, 1.0, -1.0)

    fix = _d(15, Month.January, 2025)
    f1 = s10.fixing(fix, False)
    f2 = s2.fixing(fix, False)

    tight(f1, reference_data["ssi_fix1_10y"])
    tight(f2, reference_data["ssi_fix2_2y"])
    tight(idx.fixing(fix, False), reference_data["ssi_spread_fixing"])
    exact(idx.gearing1(), reference_data["ssi_gearing1"])
    exact(idx.gearing2(), reference_data["ssi_gearing2"])


def test_swap_spread_index_spread_equals_difference() -> None:
    """Spread fixing equals g1*f1 + g2*f2 computed independently."""
    curve = _curve()
    s10 = _swap_index(10, curve)
    s2 = _swap_index(2, curve)
    idx = SwapSpreadIndex("CMS10Y-2Y", s10, s2, 1.0, -1.0)

    fix = _d(15, Month.January, 2025)
    expected = s10.fixing(fix, False) - s2.fixing(fix, False)
    tight(idx.fixing(fix, False), expected)


def test_swap_spread_index_custom_gearings() -> None:
    curve = _curve()
    s10 = _swap_index(10, curve)
    s2 = _swap_index(2, curve)
    idx = SwapSpreadIndex("CMS", s10, s2, 2.0, -0.5)

    fix = _d(15, Month.January, 2025)
    expected = 2.0 * s10.fixing(fix, False) - 0.5 * s2.fixing(fix, False)
    tight(idx.fixing(fix, False), expected)


def test_swap_spread_index_name() -> None:
    curve = _curve()
    s10 = _swap_index(10, curve)
    s2 = _swap_index(2, curve)
    idx = SwapSpreadIndex("CMS", s10, s2, 1.0, -1.0)
    # composite name with 4-decimal fixed gearings.
    assert "1.0000" in idx.name()
    assert "-1.0000" in idx.name()
    assert idx.name() == f"{s10.name()}(1.0000) + {s2.name()}(-1.0000)"


def test_swap_spread_index_inspectors() -> None:
    curve = _curve()
    s10 = _swap_index(10, curve)
    s2 = _swap_index(2, curve)
    idx = SwapSpreadIndex("CMS", s10, s2)
    assert idx.swap_index1() is s10
    assert idx.swap_index2() is s2
    assert idx.gearing1() == 1.0
    assert idx.gearing2() == -1.0
    assert idx.allows_native_fixings() is False


def test_swap_spread_index_no_maturity() -> None:
    curve = _curve()
    idx = SwapSpreadIndex("CMS", _swap_index(10, curve), _swap_index(2, curve))
    with pytest.raises(LibraryException, match="single maturity"):
        idx.maturity_date(_d(15, Month.January, 2025))


def test_swap_spread_index_mismatched_fixing_days() -> None:
    """Compatibility requirement on fixing days is enforced."""
    curve = _curve()
    euribor6m = Euribor.six_months(curve)
    s_a = SwapIndex(
        "A", Period(10, TimeUnit.Years), 2, EURCurrency(), TARGET(),
        Period(1, TimeUnit.Years), BusinessDayConvention.ModifiedFollowing,
        Thirty360(T360Convention.BondBasis), euribor6m, curve,
    )
    s_b = SwapIndex(
        "B", Period(2, TimeUnit.Years), 1, EURCurrency(), TARGET(),
        Period(1, TimeUnit.Years), BusinessDayConvention.ModifiedFollowing,
        Thirty360(T360Convention.BondBasis), euribor6m, curve,
    )
    with pytest.raises(LibraryException, match="fixing days"):
        SwapSpreadIndex("CMS", s_a, s_b)


def test_swap_spread_index_past_fixing_missing_returns_nan() -> None:
    """Missing sub-index fixing => NaN spread fixing."""
    curve = _curve()
    s10 = _swap_index(10, curve)
    s2 = _swap_index(2, curve)
    idx = SwapSpreadIndex("CMS", s10, s2)
    # no historic fixings stored -> sub-index past_fixing yields NaN
    cal = TARGET()
    fix = cal.adjust(_d(10, Month.January, 2024))
    result = idx.past_fixing(fix)
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# ProxyIbor — structural
# ---------------------------------------------------------------------------


def test_proxy_ibor_forecast() -> None:
    """forecastFixing = gearing * proxy * spread (note: multiplicative)."""
    curve = _curve()
    euribor6m = Euribor.six_months(curve)
    gearing = SimpleQuote(2.0)
    spread = SimpleQuote(1.5)
    proxy = ProxyIbor(
        "ProxyEuribor",
        Period(6, TimeUnit.Months),
        2,
        EURCurrency(),
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        False,
        Actual360(),
        gearing,
        euribor6m,
        spread,
    )
    fix = _d(15, Month.January, 2025)
    expected = 2.0 * euribor6m.fixing(fix) * 1.5
    tight(proxy.forecast_fixing(fix), expected)


def test_proxy_ibor_identity_gearing_spread() -> None:
    """gearing=1, spread=1 reproduces the proxied fixing."""
    curve = _curve()
    euribor6m = Euribor.six_months(curve)
    proxy = ProxyIbor(
        "ProxyEuribor", Period(6, TimeUnit.Months), 2, EURCurrency(), TARGET(),
        BusinessDayConvention.ModifiedFollowing, False, Actual360(),
        SimpleQuote(1.0), euribor6m, SimpleQuote(1.0),
    )
    fix = _d(15, Month.January, 2025)
    tight(proxy.forecast_fixing(fix), euribor6m.fixing(fix))


# ---------------------------------------------------------------------------
# CmsSpreadCoupon — structural shell
# ---------------------------------------------------------------------------


def test_cms_spread_coupon_construction() -> None:
    curve = _curve()
    idx = SwapSpreadIndex("CMS", _swap_index(10, curve), _swap_index(2, curve))
    cpn = CmsSpreadCoupon(
        payment_date=_d(15, Month.January, 2026),
        nominal=1.0e6,
        start_date=_d(15, Month.January, 2025),
        end_date=_d(15, Month.January, 2026),
        fixing_days=2,
        index=idx,
        gearing=1.0,
        spread=0.001,
        day_counter=Actual360(),
    )
    assert cpn.swap_spread_index() is idx
    assert cpn.nominal() == 1.0e6
    assert cpn.gearing() == 1.0
    assert cpn.spread() == 0.001


def test_cms_spread_coupon_rate_requires_pricer() -> None:
    """Without a pricer, rate() raises (pricing is a deferred carve-out)."""
    curve = _curve()
    idx = SwapSpreadIndex("CMS", _swap_index(10, curve), _swap_index(2, curve))
    cpn = CmsSpreadCoupon(
        payment_date=_d(15, Month.January, 2026),
        nominal=1.0e6,
        start_date=_d(15, Month.January, 2025),
        end_date=_d(15, Month.January, 2026),
        fixing_days=2,
        index=idx,
        day_counter=Actual360(),
    )
    with pytest.raises(LibraryException, match="pricer not set"):
        cpn.rate()


# ---------------------------------------------------------------------------
# BlackIborQuantoCouponPricer — quanto adjustment formula
# ---------------------------------------------------------------------------


def _quanto_pricer(
    caplet_vol: float,
    fx_vol: float,
    rho: float,
    vol_type: VolatilityType,
    shift: float = 0.0,
) -> BlackIborQuantoCouponPricer:
    cal = TARGET()
    dc = Actual365Fixed()
    caplet = ConstantOptionletVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=caplet_vol,
        calendar=cal,
        day_counter=dc,
        reference_date=_TODAY,
        volatility_type=vol_type,
        displacement=shift,
    )
    fxvol = BlackConstantVol(
        reference_date=_TODAY, calendar=cal, day_counter=dc, volatility=fx_vol
    )
    return BlackIborQuantoCouponPricer(fxvol, SimpleQuote(rho), caplet)


def test_quanto_adjustment_lognormal() -> None:
    """Shifted-lognormal quanto factor exp(sigma*fxsigma*rho*t1).

    Reason: hand-derived Hull quanto adjustment (Hull 6th ed., p.642).
    The pricer's caplet/fx vol surfaces are flat-constant so
    sigma=caplet_vol, fxsigma=fx_vol exactly; t1 = Act/365 from today to the
    fixing date. We feed an explicit fixing (no coupon convexity), isolating
    the quanto adjustment.
    """
    pricer = _quanto_pricer(0.20, 0.15, 0.30, VolatilityType.ShiftedLognormal)

    # bind a coupon whose fixing date is one year out.
    curve = _curve()
    euribor6m = Euribor.six_months(curve)
    start = _d(15, Month.January, 2025)
    end = _d(15, Month.July, 2025)
    cpn = IborCoupon(
        _d(17, Month.July, 2025), 1.0e6, start, end, 2, euribor6m,
        day_counter=Actual360(),
    )
    cpn.set_pricer(pricer)
    pricer.initialize(cpn)

    fixing = 0.03
    d1 = cpn.fixing_date()
    t1 = pricer.caplet_volatility().time_from_reference(d1)
    expected = 0.03 * math.exp(0.20 * 0.15 * 0.30 * t1)
    loose(pricer.quanto_adjusted_fixing(fixing), expected)


def test_quanto_adjustment_normal() -> None:
    """Normal quanto factor: fixing + sigma*fxsigma*rho*t1.

    Reason: hand-derived Hull quanto adjustment for normal caplet vols.
    """
    pricer = _quanto_pricer(0.010, 0.15, 0.30, VolatilityType.Normal)

    curve = _curve()
    euribor6m = Euribor.six_months(curve)
    start = _d(15, Month.January, 2025)
    end = _d(15, Month.July, 2025)
    cpn = IborCoupon(
        _d(17, Month.July, 2025), 1.0e6, start, end, 2, euribor6m,
        day_counter=Actual360(),
    )
    cpn.set_pricer(pricer)
    pricer.initialize(cpn)

    fixing = 0.03
    d1 = cpn.fixing_date()
    t1 = pricer.caplet_volatility().time_from_reference(d1)
    expected = 0.03 + 0.010 * 0.15 * 0.30 * t1
    loose(pricer.quanto_adjusted_fixing(fixing), expected)


def test_quanto_adjustment_identity_for_past_fixing() -> None:
    """A fixing date at/before the reference date leaves the fixing unchanged."""
    pricer = _quanto_pricer(0.20, 0.15, 0.30, VolatilityType.ShiftedLognormal)

    curve = _curve()
    euribor6m = Euribor.six_months(curve)
    # coupon starting essentially at the reference date -> fixing date <= ref.
    start = _d(15, Month.January, 2024)
    end = _d(15, Month.July, 2024)
    cpn = IborCoupon(
        _d(17, Month.July, 2024), 1.0e6, start, end, 2, euribor6m,
        day_counter=Actual360(),
    )
    cpn.set_pricer(pricer)
    pricer.initialize(cpn)
    # fixing date is before today -> no adjustment
    assert pricer.quanto_adjusted_fixing(0.03) == 0.03
