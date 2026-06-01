"""Tests for the Conundrum / Hagan CMS-coupon replication pricers (W12-A).

# C++ parity:
#   ql/cashflows/conundrumpricer.hpp + .cpp
#   ql/cashflows/cmscoupon.hpp + .cpp
#   ql/cashflows/couponpricer.hpp (CmsCouponPricer)

Cross-validates GFunctionStandard closed-form value/derivatives, the
ConundrumIntegrand value, and the AnalyticHaganPricer / NumericHaganPricer
convexity-adjusted CMS coupon rates against
``migration-harness/references/cluster/w12a.json`` — the canonical cms.cpp
conundrum reference values.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.cashflows.cms_coupon import CmsCoupon
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.conundrum_pricer import (
    AnalyticHaganPricer,
    ConundrumIntegrand,
    GFunctionFactory,
    MarketQuotedOptionPricer,
    NumericHaganPricer,
    YieldCurveModel,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.swaption.swaption_constant_vol import (
    SwaptionConstantVolatility,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_TODAY = Date.from_ymd(15, Month.January, 2024)


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/w12a")


@pytest.fixture(autouse=True)
def _eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    ObservableSettings().evaluation_date = _TODAY


def _curve(rate: float = 0.05) -> FlatForward:
    return FlatForward.from_rate(_TODAY, rate, Actual365Fixed())


def _swap_index(tenor: Period, curve: FlatForward) -> SwapIndex:
    """Reproduce the probe's 10Y EuriborSwapIsdaFixA-style index."""
    ibor = Euribor.six_months(curve)
    return SwapIndex(
        "EuriborSwapIsdaFixA",
        tenor,
        ibor.fixing_days(),
        EURCurrency(),
        ibor.fixing_calendar(),
        Period(1, TimeUnit.Years),
        BusinessDayConvention.Unadjusted,
        ibor.day_counter(),
        ibor,
    )


def _const_log_vol(v: float = 0.16) -> SwaptionConstantVolatility:
    return SwaptionConstantVolatility(
        reference_date=_TODAY,
        calendar=TARGET(),
        business_day_convention=BusinessDayConvention.Following,
        volatility=v,
        day_counter=Actual365Fixed(),
        volatility_type=VolatilityType.ShiftedLognormal,
    )


# ---------------------------------------------------------------------------
# GFunctionStandard — closed form (TIGHT)
# ---------------------------------------------------------------------------


def test_gfunction_standard_value(reference_data: dict[str, Any]) -> None:
    """GFunctionStandard value/derivatives at known x.

    TIGHT: closed-form rational function (no replication / quadrature).
    """
    g = GFunctionFactory.new_g_function_standard(1, 0.5, 10)
    x = 0.05
    tight(g(x), reference_data["gstd_value"])
    tight(g.first_derivative(x), reference_data["gstd_first"])
    tight(g.second_derivative(x), reference_data["gstd_second"])


def test_gfunction_standard_value_2(reference_data: dict[str, Any]) -> None:
    """Second sample point (q=2, delta=0.25, swapLength=5)."""
    g = GFunctionFactory.new_g_function_standard(2, 0.25, 5)
    x = 0.04
    tight(g(x), reference_data["gstd2_value"])
    tight(g.first_derivative(x), reference_data["gstd2_first"])
    tight(g.second_derivative(x), reference_data["gstd2_second"])


# ---------------------------------------------------------------------------
# ConundrumIntegrand — value at a strike (TIGHT)
# ---------------------------------------------------------------------------


def test_conundrum_integrand_value(reference_data: dict[str, Any]) -> None:
    """ConundrumIntegrand(x) = option(x) * F''(x).

    TIGHT: deterministic — Black option price * closed-form second derivative
    of F, all from the same constant-vol smile + GFunctionStandard.
    """
    curve = _curve()
    swap_index = _swap_index(Period(10, TimeUnit.Years), curve)
    start = curve.reference_date() + Period(20, TimeUnit.Years)
    payment = start + Period(1, TimeUnit.Years)
    coupon = CmsCoupon(
        payment, 1.0, start, payment, swap_index.fixing_days(), swap_index,
        1.0, 0.0, start, payment, Euribor.six_months(curve).day_counter(),
    )
    forward = swap_index.fixing(coupon.fixing_date())
    tight(forward, reference_data["ci_forward"])

    g = GFunctionFactory.new_g_function_standard(1, 0.5, 10)
    vanilla = MarketQuotedOptionPricer(
        forward, coupon.fixing_date(), swap_index.tenor(), _const_log_vol(0.16)
    )
    integrand = ConundrumIntegrand(
        vanilla, None, g, coupon.fixing_date(), payment, 4.5, forward, 0.04,
        OptionType.Call,
    )
    tight(integrand(0.06), reference_data["ci_value"])


# ---------------------------------------------------------------------------
# AnalyticHaganPricer — convexity-adjusted CMS coupon rate (LOOSE)
# ---------------------------------------------------------------------------


def _hagan_coupon(curve: FlatForward) -> tuple[CmsCoupon, SwaptionConstantVolatility]:
    swap_index = _swap_index(Period(10, TimeUnit.Years), curve)
    start = curve.reference_date() + Period(20, TimeUnit.Years)
    payment = start + Period(1, TimeUnit.Years)
    coupon = CmsCoupon(
        payment, 1.0, start, payment, swap_index.fixing_days(), swap_index,
        1.0, 0.0, start, payment, Euribor.six_months(curve).day_counter(),
    )
    return coupon, _const_log_vol(0.16)


@pytest.mark.parametrize(
    ("model", "key"),
    [
        (YieldCurveModel.Standard, "hagan_an_standard"),
        (YieldCurveModel.ExactYield, "hagan_an_exactyield"),
        (YieldCurveModel.ParallelShifts, "hagan_an_parallel"),
        (YieldCurveModel.NonParallelShifts, "hagan_an_nonparallel"),
    ],
)
def test_analytic_hagan_coupon_rate(
    reference_data: dict[str, Any], model: int, key: str
) -> None:
    """AnalyticHaganPricer convexity-adjusted CMS coupon rate vs C++.

    LOOSE: static-replication closed form (Hagan 3.4c) — the rate sits above
    the par forward by the convexity adjustment.
    """
    coupon, vol = _hagan_coupon(_curve())
    mean_rev = SimpleQuote(0.0)
    pricer = AnalyticHaganPricer(vol, model, mean_rev)
    coupon.set_pricer(pricer)
    loose(coupon.rate(), reference_data[key])


@pytest.mark.parametrize(
    ("model", "an_key", "num_key"),
    [
        (YieldCurveModel.Standard, "hagan_an_standard", "hagan_num_standard"),
        (YieldCurveModel.ExactYield, "hagan_an_exactyield", "hagan_num_exactyield"),
        (YieldCurveModel.ParallelShifts, "hagan_an_parallel", "hagan_num_parallel"),
        (
            YieldCurveModel.NonParallelShifts,
            "hagan_an_nonparallel",
            "hagan_num_nonparallel",
        ),
    ],
)
def test_numeric_hagan_coupon_rate(
    reference_data: dict[str, Any], model: int, an_key: str, num_key: str
) -> None:
    """NumericHaganPricer coupon rate vs C++, and the Numeric-vs-Analytic gap.

    LOOSE: numeric integration of the static replication. The canonical
    cms.cpp testFairRate asserts Numeric ≈ Analytic within 2e-4 *using the
    market ATM-vol matrix + SABR cube*; this probe instead uses a flat 16%
    lognormal vol for determinism, which for the shift-based G-functions
    (ParallelShifts / NonParallelShifts) exaggerates the convexity gap to
    ~2.3e-4 — and the **C++ reference shows exactly the same gap**. So we
    cross-validate the gap itself against C++ (LOOSE), confirming the Python
    static replication reproduces the C++ Numeric-vs-Analytic spread, rather
    than re-asserting the market-vol 2e-4 bound on a flat-vol setup.
    """
    coupon, vol = _hagan_coupon(_curve())
    mean_rev = SimpleQuote(0.0)
    num = NumericHaganPricer(vol, model, mean_rev)
    coupon.set_pricer(num)
    rate_num = coupon.rate()
    loose(rate_num, reference_data[num_key])

    an = AnalyticHaganPricer(vol, model, SimpleQuote(0.0))
    coupon.set_pricer(an)
    rate_an = coupon.rate()
    cpp_gap = abs(reference_data[num_key] - reference_data[an_key])
    loose(abs(rate_num - rate_an), cpp_gap)
