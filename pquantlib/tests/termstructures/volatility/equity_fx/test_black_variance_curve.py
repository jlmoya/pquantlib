"""Tests for BlackVarianceCurve — cross-validated against L2-E probe."""

from __future__ import annotations

import math

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.volatility.equity_fx.black_variance_curve import (
    BlackVarianceCurve,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_REF = reference_reader.load("cluster/l2e")
_BVC = _REF["black_variance_curve"]


def _make_curve() -> BlackVarianceCurve:
    return BlackVarianceCurve(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        dates=[
            Date.from_ymd(15, Month.September, 2026),
            Date.from_ymd(15, Month.December, 2026),
            Date.from_ymd(15, Month.June, 2027),
            Date.from_ymd(15, Month.June, 2028),
        ],
        black_vol_curve=[0.10, 0.15, 0.20, 0.25],
        day_counter=Actual365Fixed(),
    )


def test_max_date_is_last_pillar() -> None:
    c = _make_curve()
    assert c.max_date() == Date.from_ymd(15, Month.June, 2028)


def test_min_max_strike_are_infinite() -> None:
    c = _make_curve()
    assert c.min_strike() == -math.inf
    assert c.max_strike() == math.inf


def test_variances_at_pillars_match_cpp() -> None:
    c = _make_curve()
    expected_times = _BVC["times"]
    expected_vars = _BVC["variances_at_pillars"]
    for t_expected, v_expected in zip(expected_times, expected_vars, strict=True):
        tolerance.tight(c.black_variance_at_time(t_expected, 100.0), v_expected)


def test_variance_at_9mo_interpolated() -> None:
    """t=9mo lies between 6mo (t=0.501) and 1y (t=1.0); linear in variance."""
    c = _make_curve()
    # Convert 9mo to t under Actual/365 Fixed:
    t9mo = Actual365Fixed().year_fraction(
        Date.from_ymd(15, Month.June, 2026),
        Date.from_ymd(15, Month.March, 2027),
    )
    tolerance.tight(c.black_variance_at_time(t9mo, 100.0), _BVC["variance_at_9mo"])


def test_vol_at_9mo_interpolated() -> None:
    c = _make_curve()
    t9mo = Actual365Fixed().year_fraction(
        Date.from_ymd(15, Month.June, 2026),
        Date.from_ymd(15, Month.March, 2027),
    )
    tolerance.tight(c.black_vol_at_time(t9mo, 100.0), _BVC["vol_at_9mo"])


def test_curve_is_strike_independent() -> None:
    c = _make_curve()
    t9mo = Actual365Fixed().year_fraction(
        Date.from_ymd(15, Month.June, 2026),
        Date.from_ymd(15, Month.March, 2027),
    )
    tolerance.tight(c.black_variance_at_time(t9mo, 50.0), _BVC["variance_at_9mo_any_strike"])


def test_curve_rejects_mismatched_dates_vols() -> None:
    with pytest.raises(LibraryException, match="mismatch"):
        BlackVarianceCurve(
            reference_date=Date.from_ymd(15, Month.June, 2026),
            dates=[Date.from_ymd(15, Month.September, 2026)],
            black_vol_curve=[0.10, 0.15],
            day_counter=Actual365Fixed(),
        )


def test_curve_rejects_first_date_before_reference() -> None:
    with pytest.raises(LibraryException, match="dates\\[0\\] <= referenceDate"):
        BlackVarianceCurve(
            reference_date=Date.from_ymd(15, Month.June, 2026),
            dates=[Date.from_ymd(15, Month.January, 2026)],
            black_vol_curve=[0.10],
            day_counter=Actual365Fixed(),
        )


def test_curve_rejects_unsorted_dates() -> None:
    with pytest.raises(LibraryException, match="sorted unique"):
        BlackVarianceCurve(
            reference_date=Date.from_ymd(15, Month.June, 2026),
            dates=[
                Date.from_ymd(15, Month.December, 2026),
                Date.from_ymd(15, Month.September, 2026),  # earlier than prev
            ],
            black_vol_curve=[0.15, 0.10],
            day_counter=Actual365Fixed(),
        )


def test_curve_rejects_non_monotone_variance_with_force() -> None:
    """vol^2 * t must be non-decreasing if force_monotone_variance=True."""
    with pytest.raises(LibraryException, match="non-decreasing"):
        BlackVarianceCurve(
            reference_date=Date.from_ymd(15, Month.June, 2026),
            dates=[
                Date.from_ymd(15, Month.September, 2026),
                Date.from_ymd(15, Month.December, 2026),
            ],
            black_vol_curve=[0.20, 0.05],  # variance drops 2nd
            day_counter=Actual365Fixed(),
            force_monotone_variance=True,
        )


def test_curve_allows_non_monotone_variance_without_force() -> None:
    BlackVarianceCurve(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        dates=[
            Date.from_ymd(15, Month.September, 2026),
            Date.from_ymd(15, Month.December, 2026),
        ],
        black_vol_curve=[0.20, 0.05],
        day_counter=Actual365Fixed(),
        force_monotone_variance=False,
    )
