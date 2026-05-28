"""Tests for InflationTermStructure / ZeroInflationTermStructure / YoYInflationTermStructure.

Each abstract is exercised via a minimal concrete stub that hardcodes the
zero/yoy rate impl. We focus on:

- Constructor pinning + accessor parity with the C++ baseDate / frequency /
  base rate / observation lag / nominal-curve seam.
- The inflation-period bucketing rule in ``zero_rate(d)`` / ``yoy_rate(d)``.
- Range checks against ``base_date`` (not ``reference_date``).
- Seasonality install/clear via ``set_seasonality``.
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.inflation.inflation_term_structure import (
    InflationTermStructure,
)
from pquantlib.termstructures.inflation.yoy_inflation_term_structure import (
    YoYInflationTermStructure,
)
from pquantlib.termstructures.inflation.zero_inflation_term_structure import (
    ZeroInflationTermStructure,
)
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_BASE_DATE = Date.from_ymd(1, Month.January, 2020)
_REF_DATE = Date.from_ymd(31, Month.January, 2020)
_MAX_DATE = Date.from_ymd(31, Month.December, 2050)
_DC = Actual365Fixed()


class _FlatZeroCurve(ZeroInflationTermStructure):
    """A flat ``_zero_rate_impl`` for testing the abstract."""

    def __init__(self, rate: float, **kw: object) -> None:
        # All kw forwarded; rate is the constant impl value.
        kw.setdefault("base_date", _BASE_DATE)
        kw.setdefault("frequency", Frequency.Monthly)
        kw.setdefault("day_counter", _DC)
        kw.setdefault("reference_date", _REF_DATE)
        super().__init__(**kw)  # type: ignore[arg-type]
        self._rate = rate

    def _zero_rate_impl(self, t: float) -> float:
        del t
        return self._rate

    def max_date(self) -> Date:
        return _MAX_DATE


class _FlatYoYCurve(YoYInflationTermStructure):
    """A flat ``_yoy_rate_impl`` for testing the abstract."""

    def __init__(self, rate: float, **kw: object) -> None:
        kw.setdefault("base_date", _BASE_DATE)
        kw.setdefault("frequency", Frequency.Monthly)
        kw.setdefault("day_counter", _DC)
        kw.setdefault("reference_date", _REF_DATE)
        kw.setdefault("base_yoy_rate", rate)
        super().__init__(**kw)  # type: ignore[arg-type]
        self._rate = rate

    def _yoy_rate_impl(self, t: float) -> float:
        del t
        return self._rate

    def max_date(self) -> Date:
        return _MAX_DATE


# ---- accessors --------------------------------------------------------


def test_inflation_ts_pin_base_date_and_frequency() -> None:
    ts = _FlatZeroCurve(rate=0.02)
    assert ts.base_date() == _BASE_DATE
    assert ts.frequency() == Frequency.Monthly
    assert ts.reference_date() == _REF_DATE
    assert ts.max_date() == _MAX_DATE


def test_inflation_ts_observation_lag_threads_through() -> None:
    lag = Period(3, TimeUnit.Months)
    ts = _FlatZeroCurve(rate=0.02, observation_lag=lag)
    assert ts.observation_lag() == lag
    # default = None
    assert _FlatZeroCurve(rate=0.02).observation_lag() is None


def test_inflation_ts_base_rate_raises_when_absent() -> None:
    ts = _FlatZeroCurve(rate=0.02)
    assert ts.has_base_rate() is False
    with pytest.raises(LibraryException):
        ts.base_rate()


def test_yoy_ts_carries_base_rate_through_constructor() -> None:
    ts = _FlatYoYCurve(rate=0.018)
    assert ts.has_base_rate() is True
    tight(ts.base_rate(), 0.018)


def test_inflation_ts_nominal_term_structure_seam_is_none_by_default() -> None:
    ts = _FlatZeroCurve(rate=0.02)
    assert ts.nominal_term_structure() is None


# ---- zero / yoy bucketing ---------------------------------------------


def test_zero_rate_buckets_to_period_start() -> None:
    """zero_rate(d) for any d in the period must return the same value."""
    ts = _FlatZeroCurve(rate=0.02)
    z1 = ts.zero_rate(Date.from_ymd(5, Month.June, 2025))
    z2 = ts.zero_rate(Date.from_ymd(28, Month.June, 2025))
    tight(z1, 0.02)
    tight(z2, 0.02)


def test_yoy_rate_buckets_to_period_start() -> None:
    ts = _FlatYoYCurve(rate=0.025)
    y1 = ts.yoy_rate(Date.from_ymd(5, Month.June, 2025))
    y2 = ts.yoy_rate(Date.from_ymd(28, Month.June, 2025))
    tight(y1, 0.025)
    tight(y2, 0.025)


# ---- range checks pivot on base_date (not reference_date) -------------


def test_check_range_rejects_pre_base_dates() -> None:
    """zero_rate before the base date raises."""
    ts = _FlatZeroCurve(rate=0.02)
    with pytest.raises(LibraryException):
        # 2019 is before base_date 2020-01-01.
        ts.zero_rate(Date.from_ymd(15, Month.July, 2019))


def test_check_range_rejects_post_max_dates_without_extrapolate() -> None:
    ts = _FlatZeroCurve(rate=0.02)
    with pytest.raises(LibraryException):
        ts.zero_rate(Date.from_ymd(15, Month.July, 2060))
    # With extrapolate=True it should not raise.
    z = ts.zero_rate(Date.from_ymd(15, Month.July, 2060), extrapolate=True)
    tight(z, 0.02)


# ---- seasonality install ----------------------------------------------


def test_set_seasonality_to_none_clears_state() -> None:
    """Constructed without seasonality + set None is a no-op."""
    ts = _FlatZeroCurve(rate=0.02)
    assert ts.has_seasonality() is False
    ts.set_seasonality(None)
    assert ts.has_seasonality() is False
    assert ts.seasonality() is None


# ---- yoy: zero / time overloads ---------------------------------------


def test_yoy_rate_time_overload_skips_seasonality_and_bucketing() -> None:
    """yoy_rate_at_time(t) is a power-user override; flat impl returns rate."""
    ts = _FlatYoYCurve(rate=0.018)
    rate = ts.yoy_rate_at_time(2.5)
    tight(rate, 0.018)


def test_zero_rate_time_overload_skips_seasonality_and_bucketing() -> None:
    ts = _FlatZeroCurve(rate=0.02)
    rate = ts.zero_rate_at_time(3.0)
    tight(rate, 0.02)


# ---- direct abstract instantiation must fail --------------------------


def test_inflation_term_structure_abstract_cannot_instantiate() -> None:
    """InflationTermStructure has no concrete ``max_date`` so it's abstract."""
    with pytest.raises(TypeError):
        InflationTermStructure(  # type: ignore[abstract]
            base_date=_BASE_DATE,
            frequency=Frequency.Monthly,
            day_counter=_DC,
            reference_date=_REF_DATE,
        )
