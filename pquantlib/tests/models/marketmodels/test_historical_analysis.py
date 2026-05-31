"""Tests for the W9-B SequenceStatistics enabler + historical-rate analysis.

Cross-validates ``SequenceStatistics`` against
``migration-harness/references/cluster/w9b.json``; the analysis drivers are
exercised with synthetic indexes (no probe value — the C++ analysis runs over
a live historical-fixing dataset).

C++ parity:
  ql/math/statistics/sequencestatistics.hpp
  ql/models/marketmodels/historicalratesanalysis.{hpp,cpp}
  ql/models/marketmodels/historicalforwardratesanalysis.hpp
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.math.statistics.sequence_statistics import SequenceStatistics
from pquantlib.models.marketmodels.historical_forward_rates_analysis import (
    HistoricalForwardRatesAnalysis,
    historical_forward_rates_analysis,
)
from pquantlib.models.marketmodels.historical_rates_analysis import (
    HistoricalRatesAnalysis,
    historical_rates_analysis,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w9b")


# --- SequenceStatistics ------------------------------------------------------

_SAMPLES = [
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 4.0],
    [3.0, 4.0, 2.0],
    [4.0, 3.0, 5.0],
    [5.0, 6.0, 1.0],
]


def test_sequence_statistics_moments(ref: dict[str, Any]) -> None:
    stats = SequenceStatistics(3)
    for s in _SAMPLES:
        stats.add(s)
    exact(float(stats.samples()), ref["seqstat_samples"])
    exact(float(stats.size()), ref["seqstat_size"])
    m = stats.mean()
    tight(m[0], ref["seqstat_mean0"])
    tight(m[1], ref["seqstat_mean1"])
    tight(m[2], ref["seqstat_mean2"])
    v = stats.variance()
    tight(v[0], ref["seqstat_var0"])
    tight(v[2], ref["seqstat_var2"])


def test_sequence_statistics_covariance(ref: dict[str, Any]) -> None:
    stats = SequenceStatistics(3)
    for s in _SAMPLES:
        stats.add(s)
    cov = stats.covariance()
    # TIGHT: outer-product quadratic-sum + mean correction.
    tight(cov[0, 0], ref["seqstat_cov_0_0"])
    tight(cov[0, 1], ref["seqstat_cov_0_1"])
    tight(cov[0, 2], ref["seqstat_cov_0_2"])
    tight(cov[1, 2], ref["seqstat_cov_1_2"])
    tight(cov[2, 2], ref["seqstat_cov_2_2"])
    # covariance is symmetric; diagonal equals variance.
    assert np.allclose(cov, cov.T)
    v = stats.variance()
    for i in range(3):
        tight(cov[i, i], v[i])


def test_sequence_statistics_correlation(ref: dict[str, Any]) -> None:
    stats = SequenceStatistics(3)
    for s in _SAMPLES:
        stats.add(s)
    corr = stats.correlation()
    tight(corr[0, 0], ref["seqstat_corr_0_0"])
    tight(corr[0, 1], ref["seqstat_corr_0_1"])
    tight(corr[0, 2], ref["seqstat_corr_0_2"])
    tight(corr[1, 2], ref["seqstat_corr_1_2"])
    assert np.allclose(corr, corr.T)


def test_sequence_statistics_autosize() -> None:
    # dimension 0 -> auto-size on the first add.
    stats = SequenceStatistics()
    for s in _SAMPLES:
        stats.add(s)
    assert stats.size() == 3
    assert stats.samples() == 5


def test_sequence_statistics_reset() -> None:
    stats = SequenceStatistics(3)
    for s in _SAMPLES:
        stats.add(s)
    stats.reset(3)
    assert stats.samples() == 0
    # re-add and check it works again
    for s in _SAMPLES:
        stats.add(s)
    assert stats.samples() == 5


def test_sequence_statistics_dimension_mismatch() -> None:
    stats = SequenceStatistics(3)
    with pytest.raises(LibraryException):
        stats.add([1.0, 2.0])  # wrong dimension


# --- historical_rates_analysis -----------------------------------------------


def _make_ibor(name: str, tenor_months: int) -> IborIndex:
    cal = TARGET()
    return IborIndex(
        name,
        Period(tenor_months, TimeUnit.Months),
        2,
        EURCurrency(),
        cal,
        BusinessDayConvention.ModifiedFollowing,
        False,
        Actual360(),
    )


def _seed_fixings(idx: IborIndex, dates: list[Date], values: list[float]) -> None:
    for d, v in zip(dates, values, strict=True):
        idx.add_fixing(d, v, force_overwrite=True)


def test_historical_rates_analysis_pipeline() -> None:
    cal = TARGET()
    idx1 = _make_ibor("hist1", 6)
    idx2 = _make_ibor("hist2", 12)
    idx1.clear_fixings()
    idx2.clear_fixings()

    # Build a sequence of business days (weekly step) and seed fixings.
    start = Date.from_ymd(5, Month.January, 2009)
    end = Date.from_ymd(30, Month.March, 2009)
    step = Period(1, TimeUnit.Weeks)

    # Enumerate the exact dates the driver will visit, and seed fixings.
    visited: list[Date] = []
    d = cal.advance(start, 1, TimeUnit.Days, BusinessDayConvention.Following)
    while d <= end:
        visited.append(d)
        d = cal.advance_period(d, step, BusinessDayConvention.Following)

    # Deterministic, strictly-positive synthetic fixing series.
    vals1 = [0.02 + 0.001 * i for i in range(len(visited))]
    vals2 = [0.025 + 0.0008 * i for i in range(len(visited))]
    _seed_fixings(idx1, visited, vals1)
    _seed_fixings(idx2, visited, vals2)

    stats = SequenceStatistics()
    skipped: list[Date] = []
    skipped_msg: list[str] = []
    historical_rates_analysis(
        stats, skipped, skipped_msg, start, end, step, [idx1, idx2]
    )

    # No date should be skipped (all fixings present).
    assert skipped == []
    # Observations = visited - 1 (first is the baseline).
    assert stats.samples() == len(visited) - 1
    assert stats.size() == 2

    # Independently recompute the expected stats from the relative diffs.
    expected = SequenceStatistics(2)
    for i in range(1, len(visited)):
        expected.add(
            [vals1[i] / vals1[i - 1] - 1.0, vals2[i] / vals2[i - 1] - 1.0]
        )
    assert np.allclose(stats.covariance(), expected.covariance())
    assert np.allclose(stats.correlation(), expected.correlation())


def test_historical_rates_analysis_class() -> None:
    cal = TARGET()
    idx = _make_ibor("histc", 6)
    idx.clear_fixings()
    start = Date.from_ymd(5, Month.January, 2009)
    end = Date.from_ymd(2, Month.March, 2009)
    step = Period(1, TimeUnit.Weeks)
    d = cal.advance(start, 1, TimeUnit.Days, BusinessDayConvention.Following)
    visited: list[Date] = []
    while d <= end:
        visited.append(d)
        d = cal.advance_period(d, step, BusinessDayConvention.Following)
    for i, dd in enumerate(visited):
        idx.add_fixing(dd, 0.03 + 0.0005 * i, force_overwrite=True)

    stats = SequenceStatistics()
    analysis = HistoricalRatesAnalysis(stats, start, end, step, [idx])
    assert analysis.skipped_dates() == []
    assert analysis.stats().samples() == len(visited) - 1
    assert analysis.stats() is stats


def test_historical_rates_analysis_skips_missing_fixings() -> None:
    cal = TARGET()
    idx = _make_ibor("histskip", 6)
    idx.clear_fixings()
    start = Date.from_ymd(5, Month.January, 2009)
    end = Date.from_ymd(2, Month.February, 2009)
    step = Period(1, TimeUnit.Weeks)
    # Seed only a subset of dates -> the rest are skipped.
    d = cal.advance(start, 1, TimeUnit.Days, BusinessDayConvention.Following)
    visited: list[Date] = []
    while d <= end:
        visited.append(d)
        d = cal.advance_period(d, step, BusinessDayConvention.Following)
    # seed only the first two visited dates
    for i in range(2):
        idx.add_fixing(visited[i], 0.03 + 0.0005 * i, force_overwrite=True)

    stats = SequenceStatistics()
    skipped: list[Date] = []
    skipped_msg: list[str] = []
    historical_rates_analysis(
        stats, skipped, skipped_msg, start, end, step, [idx]
    )
    # The unseeded dates are skipped (each with an error message).
    assert len(skipped) == len(visited) - 2
    assert len(skipped_msg) == len(skipped)


# --- historical_forward_rates_analysis ---------------------------------------


def _deposit_index(name: str, tenor_months: int) -> IborIndex:
    """Deposit-style ibor index for curve bootstrapping."""
    return IborIndex(
        name,
        Period(tenor_months, TimeUnit.Months),
        2,
        EURCurrency(),
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        False,
        Actual360(),
    )


def _seeded_forward_curve_indexes() -> tuple[list[IborIndex], IborIndex, list[Date]]:
    """Build deposit indexes spanning to 2Y + a 3M forward index, all seeded.

    The curve spans well beyond the forward grid (1M / 4M) so every per-date
    forward read lands inside the bootstrapped curve.
    """
    cal = TARGET()
    ibors = [
        _deposit_index("fwd_d1m", 1),
        _deposit_index("fwd_d3m", 3),
        _deposit_index("fwd_d6m", 6),
        _deposit_index("fwd_d1y", 12),
        _deposit_index("fwd_d18m", 18),
        _deposit_index("fwd_d2y", 24),
    ]
    fwd = _deposit_index("fwd_fwd3m", 3)
    for ix in [*ibors, fwd]:
        ix.clear_fixings()

    start = Date.from_ymd(5, Month.January, 2009)
    end = Date.from_ymd(2, Month.March, 2009)
    step = Period(1, TimeUnit.Weeks)
    d = cal.advance(start, 1, TimeUnit.Days, BusinessDayConvention.Following)
    visited: list[Date] = []
    while d <= end:
        visited.append(d)
        d = cal.advance_period(d, step, BusinessDayConvention.Following)

    for k, dd in enumerate(visited):
        base = 0.02 + 0.0005 * k
        for j, ix in enumerate(ibors):
            ix.add_fixing(dd, base + 0.0008 * j, force_overwrite=True)
    return ibors, fwd, visited


def test_historical_forward_rates_analysis_pipeline() -> None:
    ibors, fwd, visited = _seeded_forward_curve_indexes()
    start = Date.from_ymd(5, Month.January, 2009)
    end = Date.from_ymd(2, Month.March, 2009)
    step = Period(1, TimeUnit.Weeks)

    stats = SequenceStatistics()
    skipped: list[Date] = []
    skipped_msg: list[str] = []
    failed: list[Date] = []
    failed_msg: list[str] = []
    fixing_periods: list[Period] = []
    historical_forward_rates_analysis(
        stats,
        skipped,
        skipped_msg,
        failed,
        failed_msg,
        fixing_periods,
        start,
        end,
        step,
        fwd,
        Period(1, TimeUnit.Months),
        Period(6, TimeUnit.Months),
        ibors,
        [],
        Actual360(),
    )

    # Grid = initial_gap .. horizon stepping by the 3M forward tenor -> {1M, 4M}.
    assert len(fixing_periods) == 2
    # Every date must bootstrap + read forwards cleanly (curve spans 2Y).
    assert skipped == []
    assert failed == []
    # Observations = visited - 1.
    assert stats.samples() == len(visited) - 1
    assert stats.size() == 2
    # Covariance/correlation are well-formed (symmetric, unit diagonal corr).
    cov = stats.covariance()
    assert np.allclose(cov, cov.T)
    corr = stats.correlation()
    for i in range(2):
        tight(corr[i, i], 1.0)


def test_historical_forward_rates_analysis_class_and_restores_eval_date() -> None:
    ibors, fwd, visited = _seeded_forward_curve_indexes()
    start = Date.from_ymd(5, Month.January, 2009)
    end = Date.from_ymd(2, Month.March, 2009)
    step = Period(1, TimeUnit.Weeks)

    settings = ObservableSettings()
    eval_before = settings.evaluation_date

    stats = SequenceStatistics()
    analysis = HistoricalForwardRatesAnalysis(
        stats,
        start,
        end,
        step,
        fwd,
        Period(1, TimeUnit.Months),
        Period(6, TimeUnit.Months),
        ibors,
        [],
        Actual360(),
    )
    assert analysis.failed_dates() == []
    assert analysis.skipped_dates() == []
    assert len(analysis.fixing_periods()) == 2
    assert analysis.stats().samples() == len(visited) - 1
    # The evaluation date must be restored (SavedSettings substitute).
    assert settings.evaluation_date == eval_before
