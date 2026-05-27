"""Tests for BlackVarianceSurface — cross-validated against L2-E probe."""

from __future__ import annotations

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.volatility.equity_fx.black_variance_surface import (
    BlackVarianceSurface,
    Extrapolation,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_REF = reference_reader.load("cluster/l2e")
_BVS = _REF["black_variance_surface"]

_REF_DATE = Date.from_ymd(15, Month.June, 2026)
_DATES = [
    Date.from_ymd(15, Month.September, 2026),
    Date.from_ymd(15, Month.December, 2026),
    Date.from_ymd(15, Month.June, 2027),
    Date.from_ymd(15, Month.June, 2028),
]
_STRIKES = [80.0, 100.0, 120.0]
_VOL_MATRIX = np.asarray(
    [
        [0.20, 0.21, 0.22, 0.23],
        [0.10, 0.15, 0.20, 0.25],
        [0.20, 0.21, 0.22, 0.23],
    ],
    dtype=np.float64,
)


def _make_surf() -> BlackVarianceSurface:
    return BlackVarianceSurface(
        reference_date=_REF_DATE,
        calendar=NullCalendar(),
        dates=_DATES,
        strikes=_STRIKES,
        black_vol_matrix=_VOL_MATRIX,
        day_counter=Actual365Fixed(),
    )


def test_max_date_is_last_pillar() -> None:
    s = _make_surf()
    assert s.max_date() == _DATES[-1]


def test_min_max_strike_are_grid_boundaries() -> None:
    s = _make_surf()
    assert s.min_strike() == 80.0
    assert s.max_strike() == 120.0


def test_atm_variances_at_pillars_match_cpp() -> None:
    s = _make_surf()
    expected_times = _BVS["times"]
    expected_atm = _BVS["variances_atm"]
    for t_expected, v_expected in zip(expected_times, expected_atm, strict=True):
        tolerance.tight(s.black_variance_at_time(t_expected, 100.0), v_expected)


def test_variance_at_intermediate_strike_and_time() -> None:
    """K=110 mid-strike (between 100 and 120), t=9mo mid-tenor."""
    s = _make_surf()
    t9mo = Actual365Fixed().year_fraction(_REF_DATE, Date.from_ymd(15, Month.March, 2027))
    tolerance.tight(s.black_variance_at_time(t9mo, 110.0), _BVS["variance_at_110_9mo"])


def test_vol_at_intermediate_strike_and_time() -> None:
    s = _make_surf()
    t9mo = Actual365Fixed().year_fraction(_REF_DATE, Date.from_ymd(15, Month.March, 2027))
    tolerance.tight(s.black_vol_at_time(t9mo, 110.0), _BVS["vol_at_110_9mo"])


def test_variance_at_pillar_node_is_exact() -> None:
    s = _make_surf()
    one_year = Date.from_ymd(15, Month.June, 2027)
    tolerance.tight(s.black_variance(one_year, 80.0), _BVS["variance_at_80_1y"])


def test_variance_at_pillar_at_last_date() -> None:
    s = _make_surf()
    two_year = Date.from_ymd(15, Month.June, 2028)
    tolerance.tight(s.black_variance(two_year, 100.0), _BVS["variance_at_100_2y"])


def test_variance_at_t_zero_is_zero() -> None:
    s = _make_surf()
    assert s.black_variance_at_time(0.0, 100.0) == 0.0


def test_surface_rejects_matrix_dim_mismatch() -> None:
    bad = np.asarray([[0.10, 0.15], [0.20, 0.25]], dtype=np.float64)
    with pytest.raises(LibraryException, match="vol matrix shape"):
        BlackVarianceSurface(
            reference_date=_REF_DATE,
            calendar=NullCalendar(),
            dates=_DATES,  # 4 dates
            strikes=_STRIKES,  # 3 strikes
            black_vol_matrix=bad,  # 2x2 — wrong both ways
            day_counter=Actual365Fixed(),
        )


def test_surface_rejects_first_date_before_reference() -> None:
    with pytest.raises(LibraryException, match="dates\\[0\\] < referenceDate"):
        BlackVarianceSurface(
            reference_date=_REF_DATE,
            calendar=NullCalendar(),
            dates=[Date.from_ymd(15, Month.January, 2026), *_DATES[1:]],
            strikes=_STRIKES,
            black_vol_matrix=_VOL_MATRIX,
            day_counter=Actual365Fixed(),
        )


def test_surface_constant_strike_extrapolation_clips_below_min() -> None:
    s_const = BlackVarianceSurface(
        reference_date=_REF_DATE,
        calendar=NullCalendar(),
        dates=_DATES,
        strikes=_STRIKES,
        black_vol_matrix=_VOL_MATRIX,
        day_counter=Actual365Fixed(),
        lower_extrapolation=Extrapolation.ConstantExtrapolation,
    )
    # Querying below strike[0]=80 — should clip to strike=80
    one_year = Date.from_ymd(15, Month.June, 2027)
    v_low = s_const.black_variance(one_year, 50.0, extrapolate=True)
    v_at_80 = s_const.black_variance(one_year, 80.0)
    assert v_low == v_at_80


def test_surface_constant_strike_extrapolation_clips_above_max() -> None:
    s_const = BlackVarianceSurface(
        reference_date=_REF_DATE,
        calendar=NullCalendar(),
        dates=_DATES,
        strikes=_STRIKES,
        black_vol_matrix=_VOL_MATRIX,
        day_counter=Actual365Fixed(),
        upper_extrapolation=Extrapolation.ConstantExtrapolation,
    )
    one_year = Date.from_ymd(15, Month.June, 2027)
    v_high = s_const.black_variance(one_year, 150.0, extrapolate=True)
    v_at_120 = s_const.black_variance(one_year, 120.0)
    assert v_high == v_at_120
