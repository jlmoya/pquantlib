"""Tests for the LognormalCmsSpreadPricer + CMS coupon/leg (W12-A c).

# C++ parity:
#   ql/experimental/coupons/lognormalcmsspreadpricer.hpp + .cpp
#   ql/cashflows/cmscoupon.hpp + .cpp (CmsCoupon + cms_leg)

Cross-validates the convexity-adjusted CMS-spread coupon rate against
``migration-harness/references/cluster/w12a.json``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.cashflows.cms_coupon import CmsCoupon, cms_leg
from pquantlib.cashflows.cms_spread_coupon import CmsSpreadCoupon
from pquantlib.cashflows.lognormal_cms_spread_pricer import LognormalCmsSpreadPricer
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.indexes.swap_spread_index import SwapSpreadIndex
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.conundrum_pricer import AnalyticHaganPricer, YieldCurveModel
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
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
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
# LognormalCmsSpreadPricer — CMS-spread coupon rate (LOOSE)
# ---------------------------------------------------------------------------


def _spread_coupon(curve: FlatForward) -> CmsSpreadCoupon:
    s10 = _swap_index(Period(10, TimeUnit.Years), curve)
    s2 = _swap_index(Period(2, TimeUnit.Years), curve)
    ssi = SwapSpreadIndex("CMS10Y-2Y", s10, s2, 1.0, -1.0)
    start = curve.reference_date() + Period(20, TimeUnit.Years)
    payment = start + Period(1, TimeUnit.Years)
    return CmsSpreadCoupon(
        payment, 1.0, start, payment, ssi.fixing_days(), ssi, 1.0, 0.0,
        start, payment, Euribor.six_months(curve).day_counter(),
    )


def test_cms_spread_fixings(reference_data: dict[str, Any]) -> None:
    """The two underlying swap fixings (TIGHT — par swap rates)."""
    curve = _curve()
    s10 = _swap_index(Period(10, TimeUnit.Years), curve)
    s2 = _swap_index(Period(2, TimeUnit.Years), curve)
    coupon = _spread_coupon(curve)
    fix = coupon.fixing_date()
    tight(s10.fixing(fix), reference_data["ssp_fix1_10y"])
    tight(s2.fixing(fix), reference_data["ssp_fix2_2y"])


def test_lognormal_cms_spread_rate(reference_data: dict[str, Any]) -> None:
    """LognormalCmsSpreadPricer CMS-spread coupon rate vs C++.

    LOOSE: bivariate-lognormal replication. The convexity-adjusted spread of
    the 10Y and 2Y CMS rates.
    """
    curve = _curve()
    coupon = _spread_coupon(curve)
    vol = _const_log_vol(0.16)
    cms_pricer = AnalyticHaganPricer(vol, YieldCurveModel.Standard, SimpleQuote(0.0))
    pricer = LognormalCmsSpreadPricer(cms_pricer, SimpleQuote(0.5), curve, 16)
    coupon.set_pricer(pricer)
    loose(coupon.rate(), reference_data["ssp_rate"])


def test_lognormal_cms_spread_rate_zero_corr(reference_data: dict[str, Any]) -> None:
    """At zero correlation — the swaplet rate is correlation-independent.

    LOOSE: the C++ swapletRate uses only the adjusted rates (no rho), so the
    rate equals the corr=0.5 value. Cross-validated against C++.
    """
    curve = _curve()
    coupon = _spread_coupon(curve)
    vol = _const_log_vol(0.16)
    cms_pricer = AnalyticHaganPricer(vol, YieldCurveModel.Standard, SimpleQuote(0.0))
    pricer = LognormalCmsSpreadPricer(cms_pricer, SimpleQuote(0.0), curve, 16)
    coupon.set_pricer(pricer)
    loose(coupon.rate(), reference_data["ssp_rate_corr0"])


# ---------------------------------------------------------------------------
# CmsCoupon + cms_leg — structural
# ---------------------------------------------------------------------------


def test_cms_coupon_swap_index() -> None:
    """CmsCoupon exposes its swap index + delegates rate() to the pricer."""
    curve = _curve()
    swap_index = _swap_index(Period(10, TimeUnit.Years), curve)
    start = curve.reference_date() + Period(20, TimeUnit.Years)
    payment = start + Period(1, TimeUnit.Years)
    coupon = CmsCoupon(
        payment, 1.0, start, payment, swap_index.fixing_days(), swap_index,
    )
    assert coupon.swap_index() is swap_index
    pricer = AnalyticHaganPricer(
        _const_log_vol(0.16), YieldCurveModel.Standard, SimpleQuote(0.0)
    )
    coupon.set_pricer(pricer)
    # the convexity-adjusted rate is above the par forward
    assert coupon.rate() > swap_index.fixing(coupon.fixing_date())


def test_cms_leg_builds_plain_coupons() -> None:
    """cms_leg builds a leg of CmsCoupons (uncapped)."""
    curve = _curve()
    swap_index = _swap_index(Period(10, TimeUnit.Years), curve)
    cal = TARGET()
    start = curve.reference_date() + Period(2, TimeUnit.Years)
    end = start + Period(5, TimeUnit.Years)
    schedule = Schedule.from_rule(
        start,
        end,
        Period(1, TimeUnit.Years),
        cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Forward,
        False,
    )
    leg = cms_leg(schedule, swap_index, 1.0e6)
    assert len(leg) == 5
    assert all(isinstance(cf, CmsCoupon) for cf in leg)
    assert leg[0].nominal() == 1.0e6  # type: ignore[attr-defined]


def test_cms_leg_capped_raises() -> None:
    """Capped/floored CMS legs raise (deferred to W12-B)."""
    from pquantlib.exceptions import LibraryException  # noqa: PLC0415

    curve = _curve()
    swap_index = _swap_index(Period(10, TimeUnit.Years), curve)
    cal = TARGET()
    start = curve.reference_date() + Period(2, TimeUnit.Years)
    end = start + Period(5, TimeUnit.Years)
    schedule = Schedule.from_rule(
        start,
        end,
        Period(1, TimeUnit.Years),
        cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Forward,
        False,
    )
    with pytest.raises(LibraryException, match="capped/floored CMS legs"):
        cms_leg(schedule, swap_index, 1.0e6, caps=0.05)
