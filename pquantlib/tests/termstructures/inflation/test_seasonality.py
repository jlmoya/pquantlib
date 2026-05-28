"""Tests for MultiplicativePriceSeasonality.

Cross-validates the seasonality-factor lookup against
``migration-harness/references/l7a/foundations.json`` (seasonality block).
Correction math is verified against an independent re-derivation of the
C++ ``seasonalityCorrection`` formula.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.inflation.seasonality import (
    MultiplicativePriceSeasonality,
    Seasonality,
)
from pquantlib.termstructures.inflation.zero_inflation_term_structure import (
    ZeroInflationTermStructure,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return load_reference("l7a/foundations")


_FACTORS = [1.0, 1.005, 1.01, 1.015, 1.02, 1.025, 1.0, 0.995, 0.99, 0.985, 0.98, 0.975]
_BASE_DATE = Date.from_ymd(1, Month.January, 2020)


def _stationary_monthly() -> MultiplicativePriceSeasonality:
    return MultiplicativePriceSeasonality(_BASE_DATE, Frequency.Monthly, _FACTORS)


# ---- accessor pinning -----------------------------------------------


def test_inspectors_round_trip() -> None:
    s = _stationary_monthly()
    assert s.seasonality_base_date() == _BASE_DATE
    assert s.frequency() == Frequency.Monthly
    assert s.seasonality_factors() == _FACTORS
    # mutation guard: the returned list is a copy.
    s.seasonality_factors()[0] = 99.0
    assert s.seasonality_factors()[0] == 1.0


# ---- factor lookup against the probe -------------------------------


def test_factor_lookup_matches_probe_samples(reference: dict[str, Any]) -> None:
    """For each (date, factor) sample in the probe, the Python factor must match."""
    s = _stationary_monthly()
    block = reference["seasonality"]
    assert block["base_date_serial"] == _BASE_DATE.serial
    assert block["frequency"] == int(Frequency.Monthly)
    for sample in block["samples"]:
        d = Date(serial=sample["date_serial"])
        factor = s.seasonality_factor(d)
        tight(factor, sample["factor"])


# ---- validation -----------------------------------------------------


def test_validate_rejects_factor_count_mismatch() -> None:
    """A factor list whose length isn't a multiple of frequency raises."""
    with pytest.raises(LibraryException):
        MultiplicativePriceSeasonality(_BASE_DATE, Frequency.Monthly, _FACTORS[:5])


def test_validate_rejects_unsupported_frequency() -> None:
    """Annual frequency is not in the allowed set (semiannual..daily)."""
    with pytest.raises(LibraryException):
        MultiplicativePriceSeasonality(_BASE_DATE, Frequency.Annual, [1.0])


def test_validate_rejects_empty_factors() -> None:
    with pytest.raises(LibraryException):
        MultiplicativePriceSeasonality(_BASE_DATE, Frequency.Monthly, [])


# ---- is_consistent --------------------------------------------------


class _FlatZeroCurve(ZeroInflationTermStructure):
    """Minimal concrete to test is_consistent + correct_zero_rate."""

    def __init__(self, rate: float) -> None:
        super().__init__(
            base_date=Date.from_ymd(1, Month.January, 2020),
            frequency=Frequency.Monthly,
            day_counter=Actual365Fixed(),
            reference_date=Date.from_ymd(1, Month.February, 2020),
        )
        self._rate = rate

    def _zero_rate_impl(self, t: float) -> float:
        del t
        return self._rate

    def max_date(self) -> Date:
        return Date.from_ymd(31, Month.December, 2050)


def test_is_consistent_stationary_passes() -> None:
    s = _stationary_monthly()
    assert s.is_consistent(_FlatZeroCurve(rate=0.02)) is True


def test_is_consistent_multi_year_stationary_repeat_passes() -> None:
    """24 monthly factors that repeat exactly each year are consistent."""
    two_year = _FACTORS + _FACTORS  # 24 factors, year 2 == year 1.
    s = MultiplicativePriceSeasonality(_BASE_DATE, Frequency.Monthly, two_year)
    assert s.is_consistent(_FlatZeroCurve(rate=0.02)) is True


# ---- correct_zero_rate end-to-end -----------------------------------


def test_correct_zero_rate_with_factor_one_is_identity_when_base_anchor_matches() -> None:
    """If both factor_at and factor_base = 1.0, the corrected rate equals the input."""
    s = _stationary_monthly()
    # Use a date whose factor = 1.0 (January = factor[0]).
    ts = _FlatZeroCurve(rate=0.02)
    # ts.base_date = 2020-01-01 → factor = 1.0; pick d = 2021-01-15 → factor = 1.0 too.
    d = Date.from_ymd(15, Month.January, 2021)
    corrected = s.correct_zero_rate(d, 0.02, ts)
    tight(corrected, 0.02)


def test_correct_yoy_rate_factor_ratio_drives_correction() -> None:
    """For YoY: f = factor(d) / factor(d - 1y); corrected = (r+1)*f - 1."""
    s = _stationary_monthly()
    ts = _FlatZeroCurve(rate=0.02)
    # d = 2021-06-15 → factor(June) = 1.025; factor(2020-06-15) = 1.025; ratio = 1.
    d = Date.from_ymd(15, Month.June, 2021)
    factor_at = s.seasonality_factor(d)
    factor_1y_before = s.seasonality_factor(d - Period(1, TimeUnit.Years))
    f = factor_at / factor_1y_before
    expected = (0.02 + 1.0) * f - 1.0
    corrected = s.correct_yoy_rate(d, 0.02, ts)
    tight(corrected, expected)


def test_correct_zero_rate_nontrivial_factor() -> None:
    """A date whose factor differs from base produces a non-trivial correction."""
    s = _stationary_monthly()
    ts = _FlatZeroCurve(rate=0.02)
    # d = 2021-06-15 → factor(June) = 1.025; factor(2020-01-01) = 1.0 (base).
    d = Date.from_ymd(15, Month.June, 2021)
    corrected = s.correct_zero_rate(d, 0.02, ts)
    # Re-derive locally:
    # factor_at = 1.025; factor_base = 1.0; seasonalityAt = 1.025
    # period_start = 2021-06-01; t = year_fraction(2020-01-01, 2021-06-01) (Act/365F)
    # f = 1.025 ** (1 / t)
    factor_at = s.seasonality_factor(d)
    factor_base = s.seasonality_factor(_BASE_DATE)
    seasonality_at = factor_at / factor_base
    period_start = Date.from_ymd(1, Month.June, 2021)
    t = ts.day_counter().year_fraction(_BASE_DATE, period_start)
    f = seasonality_at ** (1.0 / t)
    expected = (0.02 + 1.0) * f - 1.0
    tight(corrected, expected)


# ---- abstract base class --------------------------------------------


def test_seasonality_abstract_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Seasonality()  # type: ignore[abstract]


def test_seasonality_default_is_consistent_is_true() -> None:
    """Default ``is_consistent`` on the base returns True for any ts."""
    # Use the concrete to exercise the inherited default — overriding it.
    s = _stationary_monthly()
    # Default base impl returns True; the concrete overrides it. Just sanity:
    assert Seasonality.is_consistent(s, _FlatZeroCurve(rate=0.02)) is True
