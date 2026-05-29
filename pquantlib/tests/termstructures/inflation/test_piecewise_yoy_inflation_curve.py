"""Tests for PiecewiseYoYInflationCurve + YearOnYearInflationSwapHelper.

# C++ parity: ql/termstructures/inflation/piecewiseyoyinflationcurve.hpp +
   ql/termstructures/inflation/inflationhelpers.{hpp,cpp} (v1.42.1).

Roundtrip: with the YoY helper's simplified `implied_quote = ts.yoy_rate(
maturity - lag)` (documented divergence in inflation_helpers.py), the
bootstrap pins each pillar's YoY rate to the input quote *exactly* (under
LinearInterpolation the bootstrap solves for ``data[i]`` such that the
curve's yoy_rate at the pillar matches the quote).

These are synthetic-roundtrip tests — they don't probe a C++ value
(the C++ helper builds a full YYIIS over a multi-coupon schedule;
porting that requires a YoY-coupon leg builder + pricer, which
is deferred per the L8-A divergence note).
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.inflation.eu_hicp import YoYEUHICP
from pquantlib.termstructures.inflation.inflation_helpers import (
    YearOnYearInflationSwapHelper,
)
from pquantlib.termstructures.inflation.piecewise_yoy_inflation_curve import (
    PiecewiseYoYInflationCurve,
)
from pquantlib.testing.tolerance import loose
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def _build_yyii() -> YoYEUHICP:
    """EUHICP YoY index — no past fixings needed for the YoY helper path."""
    return YoYEUHICP()


def test_piecewise_yoy_roundtrip_matches_input_quotes() -> None:
    """Each helper.implied_quote() ≈ input quote (LOOSE).

    Synthetic test (no C++ probe — see module docstring).
    """
    today = Date.from_ymd(15, Month.January, 2020)
    quotes = [0.020, 0.025, 0.030]
    maturities = [
        Date.from_ymd(15, Month.January, 2022),
        Date.from_ymd(15, Month.January, 2025),
        Date.from_ymd(15, Month.January, 2030),
    ]
    yyii = _build_yyii()
    calendar = TARGET()
    bdc = BusinessDayConvention.ModifiedFollowing
    swap_dc = Thirty360(Thirty360Convention.BondBasis)
    obs_lag = Period(3, TimeUnit.Months)

    helpers = [
        YearOnYearInflationSwapHelper(
            quote=q,
            observation_lag=obs_lag,
            maturity=m,
            calendar=calendar,
            payment_convention=bdc,
            day_counter=swap_dc,
            index=yyii,
        )
        for q, m in zip(quotes, maturities, strict=True)
    ]

    curve = PiecewiseYoYInflationCurve(
        reference_date=today,
        calendar=calendar,
        day_counter=Actual360(),
        observation_lag=obs_lag,
        frequency=Frequency.Monthly,
        base_rate=0.018,
        instruments=helpers,
    )
    assert curve is not None
    for h, q in zip(helpers, quotes, strict=True):
        loose(
            h.implied_quote(),
            q,
            reason="bootstrap solver tolerance — target accuracy 1e-12.",
        )


def test_piecewise_yoy_curve_base_date_matches_lag_rule() -> None:
    today = Date.from_ymd(15, Month.January, 2020)
    yyii = _build_yyii()
    helper = YearOnYearInflationSwapHelper(
        quote=0.022,
        observation_lag=Period(3, TimeUnit.Months),
        maturity=Date.from_ymd(15, Month.January, 2022),
        calendar=TARGET(),
        payment_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=Thirty360(Thirty360Convention.BondBasis),
        index=yyii,
    )
    curve = PiecewiseYoYInflationCurve(
        reference_date=today,
        calendar=TARGET(),
        day_counter=Actual360(),
        observation_lag=Period(3, TimeUnit.Months),
        frequency=Frequency.Monthly,
        base_rate=0.018,
        instruments=[helper],
    )
    # # C++ parity: base_date = inflation_period(today - 3M, Monthly).first =
    # inflation_period(Oct-15-2019, Monthly).first = Oct-1-2019.
    assert curve.base_date() == Date.from_ymd(1, Month.October, 2019)


def test_piecewise_yoy_requires_helpers() -> None:
    today = Date.from_ymd(15, Month.January, 2020)
    with pytest.raises(Exception, match="no helpers"):
        PiecewiseYoYInflationCurve(
            reference_date=today,
            calendar=TARGET(),
            day_counter=Actual360(),
            observation_lag=Period(3, TimeUnit.Months),
            frequency=Frequency.Monthly,
            base_rate=0.018,
            instruments=[],
        )


def test_piecewise_yoy_base_rate_preserved_under_traits() -> None:
    """YoY traits never overwrite ``data[0]``.

    Verifies the key trait-difference between zero and YoY traits: the
    user-supplied base rate is preserved through the bootstrap.
    """
    today = Date.from_ymd(15, Month.January, 2020)
    base_rate = 0.018
    yyii = _build_yyii()
    helpers = [
        YearOnYearInflationSwapHelper(
            quote=0.025,
            observation_lag=Period(3, TimeUnit.Months),
            maturity=Date.from_ymd(15, Month.January, 2025),
            calendar=TARGET(),
            payment_convention=BusinessDayConvention.ModifiedFollowing,
            day_counter=Thirty360(Thirty360Convention.BondBasis),
            index=yyii,
        )
    ]
    curve = PiecewiseYoYInflationCurve(
        reference_date=today,
        calendar=TARGET(),
        day_counter=Actual360(),
        observation_lag=Period(3, TimeUnit.Months),
        frequency=Frequency.Monthly,
        base_rate=base_rate,
        instruments=helpers,
    )
    # data[0] should still hold the user-supplied base_rate.
    assert curve.data()[0] == base_rate
