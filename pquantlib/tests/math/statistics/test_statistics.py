"""Cross-validate GeneralStatistics + IncrementalStatistics against the C++ probe.

Probe key: cluster/b -> "statistics" -> {general, incremental}.

The probe feeds samples 1..10 (uniform weights) to both aggregators and
records mean/variance/stddev/skewness/kurtosis/min/max/samples.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.statistics.general_statistics import GeneralStatistics
from pquantlib.math.statistics.incremental_statistics import IncrementalStatistics
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/b")


@pytest.fixture
def samples() -> list[float]:
    """Samples 1..10 — mirrors the C++ probe input vector."""
    return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


# --- GeneralStatistics --------------------------------------------------


def test_general_statistics_matches_cpp(cpp: dict[str, Any], samples: list[float]) -> None:
    gs = GeneralStatistics()
    for v in samples:
        gs.add(v)
    block = cpp["statistics"]["general"]
    tolerance.tight(gs.mean(), float(block["mean"]))
    tolerance.tight(gs.variance(), float(block["variance"]))
    tolerance.tight(gs.standard_deviation(), float(block["standardDeviation"]))
    tolerance.tight(gs.skewness(), float(block["skewness"]))
    tolerance.tight(gs.kurtosis(), float(block["kurtosis"]))
    tolerance.exact(gs.min(), float(block["min"]))
    tolerance.exact(gs.max(), float(block["max"]))
    assert gs.samples() == int(block["samples"])


def test_general_statistics_empty_rejects_mean() -> None:
    gs = GeneralStatistics()
    with pytest.raises(LibraryException):
        gs.mean()


def test_general_statistics_single_sample_rejects_variance() -> None:
    gs = GeneralStatistics()
    gs.add(1.0)
    with pytest.raises(LibraryException):
        gs.variance()


def test_general_statistics_reset_clears_samples(samples: list[float]) -> None:
    gs = GeneralStatistics()
    for v in samples:
        gs.add(v)
    assert gs.samples() == 10
    gs.reset()
    assert gs.samples() == 0


def test_general_statistics_weighted_mean() -> None:
    gs = GeneralStatistics()
    gs.add(1.0, weight=2.0)
    gs.add(3.0, weight=2.0)
    # weighted mean = (2*1 + 2*3) / 4 = 2.0
    tolerance.tight(gs.mean(), 2.0)


def test_general_statistics_negative_weight_rejected() -> None:
    gs = GeneralStatistics()
    with pytest.raises(LibraryException):
        gs.add(1.0, weight=-1.0)


def test_general_statistics_percentile(samples: list[float]) -> None:
    gs = GeneralStatistics()
    for v in samples:
        gs.add(v)
    # 50th percentile with uniform weights and ten samples: weight target
    # is 0.5 * 10 = 5; cumulative weight reaches 5 at the 5th sample (5.0).
    tolerance.exact(gs.percentile(0.5), 5.0)


# --- IncrementalStatistics ----------------------------------------------


def test_incremental_statistics_matches_cpp(cpp: dict[str, Any], samples: list[float]) -> None:
    is_ = IncrementalStatistics()
    for v in samples:
        is_.add(v)
    block = cpp["statistics"]["incremental"]
    tolerance.tight(is_.mean(), float(block["mean"]))
    tolerance.tight(is_.variance(), float(block["variance"]))
    tolerance.tight(is_.standard_deviation(), float(block["standardDeviation"]))
    tolerance.exact(is_.min(), float(block["min"]))
    tolerance.exact(is_.max(), float(block["max"]))


def test_incremental_statistics_samples_count(samples: list[float]) -> None:
    is_ = IncrementalStatistics()
    for v in samples:
        is_.add(v)
    assert is_.samples() == 10


def test_incremental_statistics_reset(samples: list[float]) -> None:
    is_ = IncrementalStatistics()
    for v in samples:
        is_.add(v)
    is_.reset()
    assert is_.samples() == 0


def test_incremental_statistics_empty_mean_rejects() -> None:
    is_ = IncrementalStatistics()
    with pytest.raises(LibraryException):
        is_.mean()


def test_incremental_statistics_downside_variance() -> None:
    is_ = IncrementalStatistics()
    for v in [-2.0, -1.0, 1.0, 2.0]:
        is_.add(v)
    # Two negative samples (-1, -2): weighted-second-moment / weight_sum =
    # ((1)*1 + (1)*4) / 2 = 2.5; r1 = N/(N-1) = 2/1 = 2. → 5.0.
    tolerance.tight(is_.downside_variance(), 5.0)


def test_incremental_statistics_negative_weight_rejected() -> None:
    is_ = IncrementalStatistics()
    with pytest.raises(LibraryException):
        is_.add(1.0, weight=-1.0)
