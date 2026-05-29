"""Tests for ZeroCouponInflationSwapHelper + YearOnYearInflationSwapHelper.

# C++ parity: ql/termstructures/inflation/inflationhelpers.{hpp,cpp}
   (v1.42.1).

Unit-level checks for the helper API: constructor, pillar date, quote
mechanics, and ``implied_quote`` after binding a curve. Bootstrap
roundtrip tests live in ``test_piecewise_*_inflation_curve.py``.
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.inflation.eu_hicp import EUHICP, YoYEUHICP
from pquantlib.indexes.inflation.inflation_index import inflation_period
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.inflation.inflation_helpers import (
    YearOnYearInflationSwapHelper,
    ZeroCouponInflationSwapHelper,
)
from pquantlib.termstructures.inflation.interpolated_yoy_inflation_curve import (
    InterpolatedYoYInflationCurve,
)
from pquantlib.termstructures.inflation.interpolated_zero_inflation_curve import (
    InterpolatedZeroInflationCurve,
)
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

# ---- ZeroCouponInflationSwapHelper ----------------------------------


def test_zero_helper_pillar_date_matches_cpp() -> None:
    """C++ parity: pillar = inflation_period(maturity - lag, freq).first."""
    maturity = Date.from_ymd(15, Month.January, 2022)
    lag = Period(3, TimeUnit.Months)
    helper = ZeroCouponInflationSwapHelper(
        quote=0.022,
        observation_lag=lag,
        maturity=maturity,
        calendar=TARGET(),
        payment_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=Thirty360(Thirty360Convention.BondBasis),
        index=EUHICP(),
    )
    expected, _ = inflation_period(maturity - lag, Frequency.Monthly)
    assert helper.pillar_date() == expected


def test_zero_helper_quote_wraps_float() -> None:
    """Float quotes are wrapped in a SimpleQuote."""
    h = ZeroCouponInflationSwapHelper(
        quote=0.025,
        observation_lag=Period(3, TimeUnit.Months),
        maturity=Date.from_ymd(15, Month.January, 2025),
        calendar=TARGET(),
        payment_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=Thirty360(Thirty360Convention.BondBasis),
        index=EUHICP(),
    )
    tight(h.quote().value(), 0.025)


def test_zero_helper_quote_handle_passthrough() -> None:
    """A ``Quote`` instance is used directly (no wrapping)."""
    q = SimpleQuote(0.022)
    h = ZeroCouponInflationSwapHelper(
        quote=q,
        observation_lag=Period(3, TimeUnit.Months),
        maturity=Date.from_ymd(15, Month.January, 2025),
        calendar=TARGET(),
        payment_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=Thirty360(Thirty360Convention.BondBasis),
        index=EUHICP(),
    )
    assert h.quote() is q


def test_zero_helper_implied_quote_requires_set_term_structure() -> None:
    """Calling implied_quote before set_term_structure raises."""
    h = ZeroCouponInflationSwapHelper(
        quote=0.022,
        observation_lag=Period(3, TimeUnit.Months),
        maturity=Date.from_ymd(15, Month.January, 2025),
        calendar=TARGET(),
        payment_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=Thirty360(Thirty360Convention.BondBasis),
        index=EUHICP(),
    )
    with pytest.raises(LibraryException, match="term structure not set"):
        h.implied_quote()


def test_zero_helper_observation_lag_inspector() -> None:
    lag = Period(2, TimeUnit.Months)
    h = ZeroCouponInflationSwapHelper(
        quote=0.022,
        observation_lag=lag,
        maturity=Date.from_ymd(15, Month.January, 2025),
        calendar=TARGET(),
        payment_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=Thirty360(Thirty360Convention.BondBasis),
        index=EUHICP(),
    )
    assert h.observation_lag() == lag


def test_zero_helper_inflation_index_inspector() -> None:
    zii = EUHICP()
    h = ZeroCouponInflationSwapHelper(
        quote=0.022,
        observation_lag=Period(3, TimeUnit.Months),
        maturity=Date.from_ymd(15, Month.January, 2025),
        calendar=TARGET(),
        payment_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=Thirty360(Thirty360Convention.BondBasis),
        index=zii,
    )
    assert h.inflation_index() is zii


def test_zero_helper_swap_built_on_set_term_structure() -> None:
    """``swap()`` returns the synthetic ZCIIS after binding the curve."""
    today = Date.from_ymd(15, Month.January, 2020)
    zii = EUHICP()
    zii.add_fixing(Date.from_ymd(1, Month.October, 2019), 100.0, force_overwrite=True)
    h = ZeroCouponInflationSwapHelper(
        quote=0.022,
        observation_lag=Period(3, TimeUnit.Months),
        maturity=Date.from_ymd(15, Month.January, 2025),
        calendar=TARGET(),
        payment_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=Thirty360(Thirty360Convention.BondBasis),
        index=zii,
    )
    assert h.swap() is None
    curve = InterpolatedZeroInflationCurve(
        reference_date=today,
        dates=[Date.from_ymd(1, Month.October, 2019), Date.from_ymd(1, Month.January, 2026)],
        rates=[0.02, 0.025],
        frequency=Frequency.Monthly,
        day_counter=Actual360(),
    )
    h.set_term_structure(curve)
    assert h.swap() is not None


# ---- YearOnYearInflationSwapHelper ---------------------------------


def test_yoy_helper_pillar_date_matches_cpp() -> None:
    maturity = Date.from_ymd(15, Month.January, 2025)
    lag = Period(3, TimeUnit.Months)
    h = YearOnYearInflationSwapHelper(
        quote=0.025,
        observation_lag=lag,
        maturity=maturity,
        calendar=TARGET(),
        payment_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=Thirty360(Thirty360Convention.BondBasis),
        index=YoYEUHICP(),
    )
    expected, _ = inflation_period(maturity - lag, Frequency.Monthly)
    assert h.pillar_date() == expected


def test_yoy_helper_implied_quote_requires_set_term_structure() -> None:
    h = YearOnYearInflationSwapHelper(
        quote=0.022,
        observation_lag=Period(3, TimeUnit.Months),
        maturity=Date.from_ymd(15, Month.January, 2025),
        calendar=TARGET(),
        payment_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=Thirty360(Thirty360Convention.BondBasis),
        index=YoYEUHICP(),
    )
    with pytest.raises(LibraryException, match="term structure not set"):
        h.implied_quote()


def test_yoy_helper_implied_quote_after_binding_matches_forecast() -> None:
    """After bind, implied_quote() == ts.yoy_rate(maturity - lag).

    Verifies the simplified divergence: under flat-zero discounting the
    YYIIS fair rate reduces to the YoY forecast at the pillar.
    """
    today = Date.from_ymd(15, Month.January, 2020)
    yyii = YoYEUHICP()
    maturity = Date.from_ymd(15, Month.January, 2025)
    lag = Period(3, TimeUnit.Months)
    h = YearOnYearInflationSwapHelper(
        quote=0.025,
        observation_lag=lag,
        maturity=maturity,
        calendar=TARGET(),
        payment_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=Thirty360(Thirty360Convention.BondBasis),
        index=yyii,
    )
    curve = InterpolatedYoYInflationCurve(
        reference_date=today,
        dates=[Date.from_ymd(1, Month.October, 2019), Date.from_ymd(1, Month.January, 2026)],
        rates=[0.018, 0.025],
        frequency=Frequency.Monthly,
        day_counter=Actual360(),
    )
    h.set_term_structure(curve)
    expected = curve.yoy_rate(maturity - lag, True)
    tight(h.implied_quote(), expected)


def test_yoy_helper_inflation_index_inspector() -> None:
    yyii = YoYEUHICP()
    h = YearOnYearInflationSwapHelper(
        quote=0.022,
        observation_lag=Period(3, TimeUnit.Months),
        maturity=Date.from_ymd(15, Month.January, 2025),
        calendar=TARGET(),
        payment_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=Thirty360(Thirty360Convention.BondBasis),
        index=yyii,
    )
    assert h.inflation_index() is yyii
