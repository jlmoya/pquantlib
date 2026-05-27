"""Tests for the BlackVolTermStructure abstract bases (equity / FX)."""

from __future__ import annotations

import math

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVarianceTermStructure,
    BlackVolatilityTermStructure,
    BlackVolTermStructure,
)
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# --- minimal concrete stubs --------------------------------------------------


class _StubBlackVol(BlackVolatilityTermStructure):
    """Adapter subclass: only _black_vol_impl needs implementing."""

    def max_date(self) -> Date:
        return Date.from_ymd(31, Month.December, 2050)

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    def _black_vol_impl(self, t: float, strike: float) -> float:
        _ = t, strike
        return 0.20


class _StubBlackVariance(BlackVarianceTermStructure):
    """Adapter subclass: only _black_variance_impl needs implementing."""

    def max_date(self) -> Date:
        return Date.from_ymd(31, Month.December, 2050)

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    def _black_variance_impl(self, t: float, strike: float) -> float:
        _ = strike
        # variance = 0.04 * t (so vol = 0.20 across all (t, strike))
        return 0.04 * t


# --- abstract instantiation guard --------------------------------------------


def test_cannot_instantiate_abstract_root() -> None:
    with pytest.raises(TypeError):
        BlackVolTermStructure()  # type: ignore[abstract]


def test_cannot_instantiate_abstract_vol_adapter() -> None:
    with pytest.raises(TypeError):
        BlackVolatilityTermStructure()  # type: ignore[abstract]


def test_cannot_instantiate_abstract_variance_adapter() -> None:
    with pytest.raises(TypeError):
        BlackVarianceTermStructure()  # type: ignore[abstract]


# --- default business-day convention -----------------------------------------


def test_default_business_day_convention_is_following() -> None:
    ts = _StubBlackVol(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    assert ts.business_day_convention() == BusinessDayConvention.Following


# --- BlackVolatilityTermStructure: vol → variance ---------------------------


def test_vol_adapter_derives_variance_from_vol() -> None:
    ts = _StubBlackVol(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    # variance(t=1, K=100) = vol^2 * t = 0.04
    tolerance.tight(ts.black_variance_at_time(1.0, 100.0), 0.04)


# --- BlackVarianceTermStructure: variance → vol -----------------------------


def test_variance_adapter_derives_vol_from_variance() -> None:
    ts = _StubBlackVariance(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    # vol = sqrt(variance(t=1) / t) = sqrt(0.04) = 0.20
    tolerance.tight(ts.black_vol_at_time(1.0, 100.0), 0.20)


def test_variance_adapter_uses_floor_at_t_zero() -> None:
    """C++ ``BlackVarianceTermStructure::blackVolImpl`` uses 1e-5 epsilon at t=0."""
    ts = _StubBlackVariance(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    # variance(t=1e-5) = 0.04 * 1e-5 → vol = sqrt(0.04) = 0.20
    tolerance.tight(ts.black_vol_at_time(0.0, 100.0), 0.20)


# --- date-anchored API ------------------------------------------------------


def test_black_vol_by_date_matches_time() -> None:
    ref = Date.from_ymd(15, Month.June, 2026)
    one_year = Date.from_ymd(15, Month.June, 2027)
    ts = _StubBlackVol(reference_date=ref, calendar=NullCalendar(), day_counter=Actual365Fixed())
    # Same as time-anchored: vol is constant 0.20
    tolerance.tight(ts.black_vol(one_year, 100.0), 0.20)


def test_black_variance_by_date_matches_time() -> None:
    ref = Date.from_ymd(15, Month.June, 2026)
    one_year = Date.from_ymd(15, Month.June, 2027)
    ts = _StubBlackVariance(reference_date=ref, calendar=NullCalendar(), day_counter=Actual365Fixed())
    # variance(1y) = 0.04
    tolerance.tight(ts.black_variance(one_year, 100.0), 0.04)


# --- forward variance / vol -------------------------------------------------


def test_forward_variance_between_two_times() -> None:
    ts = _StubBlackVariance(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    # variance(2) - variance(1) = 0.08 - 0.04 = 0.04
    tolerance.tight(ts.black_forward_variance_at_time(1.0, 2.0, 100.0), 0.04)


def test_forward_vol_between_two_times() -> None:
    ts = _StubBlackVariance(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    # vol_fwd = sqrt((v2 - v1) / (t2 - t1)) = sqrt(0.04 / 1) = 0.20
    tolerance.tight(ts.black_forward_vol_at_time(1.0, 2.0, 100.0), 0.20)


def test_forward_vol_zero_window_at_zero() -> None:
    ts = _StubBlackVariance(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    # time1 == time2 == 0: epsilon = 1e-5, var(1e-5) / 1e-5 → vol = 0.20
    tolerance.tight(ts.black_forward_vol_at_time(0.0, 0.0, 100.0), 0.20)


def test_forward_vol_zero_window_at_nonzero() -> None:
    ts = _StubBlackVariance(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    # time1 == time2 == 1.0: central FD, epsilon = 1e-5. The finite
    # difference introduces ~O(epsilon^2) rounding even for an exactly
    # linear variance impl, so LOOSE tier (1e-8) is appropriate here.
    tolerance.loose(ts.black_forward_vol_at_time(1.0, 1.0, 100.0), 0.20)


def test_forward_vol_rejects_date1_after_date2() -> None:
    ref = Date.from_ymd(15, Month.June, 2026)
    ts = _StubBlackVariance(reference_date=ref, calendar=NullCalendar(), day_counter=Actual365Fixed())
    with pytest.raises(LibraryException, match="later than"):
        ts.black_forward_vol(
            Date.from_ymd(15, Month.June, 2028), Date.from_ymd(15, Month.June, 2027), 100.0
        )


def test_forward_variance_rejects_date1_after_date2() -> None:
    ref = Date.from_ymd(15, Month.June, 2026)
    ts = _StubBlackVariance(reference_date=ref, calendar=NullCalendar(), day_counter=Actual365Fixed())
    with pytest.raises(LibraryException, match="later than"):
        ts.black_forward_variance(
            Date.from_ymd(15, Month.June, 2028), Date.from_ymd(15, Month.June, 2027), 100.0
        )
