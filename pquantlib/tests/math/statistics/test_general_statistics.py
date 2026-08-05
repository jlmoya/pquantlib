"""Cross-validate GeneralStatistics against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/statistics.json`` — the
five data-set blocks, each carrying its own inputs so the cases are walked
rather than restated.

The interesting part is the quantile convention: ``percentile`` walks the
*weight* integral over the sorted samples and stops at the first sample
whose cumulative weight reaches ``y * weightSum()``. The ``ties`` block has
repeated values and a 0.5 target that lands exactly on a cumulative-weight
boundary; the ``weighted`` block has tied values carrying different weights.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.statistics.general_statistics import NULL_REAL, GeneralStatistics
from pquantlib.math.statistics.statistics import Statistics
from pquantlib.testing import reference_reader, tolerance

DATASETS = ("ties", "spread", "positive", "one_below", "weighted")


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/statistics")


def build(block: dict[str, Any]) -> Statistics:
    """Rebuild the C++ accumulator described by a data-set block."""
    stats = Statistics()
    weights = [float(w) for w in block["weights"]]
    stats.add_sequence([float(v) for v in block["data"]], weights or None)
    return stats


@pytest.mark.parametrize("name", DATASETS)
def test_sample_count(cpp: dict[str, Any], name: str) -> None:
    """Sample count matches C++ exactly."""
    block = cpp[name]
    assert build(block).samples() == int(block["samples"])


@pytest.mark.parametrize("name", DATASETS)
def test_weight_sum_min_max_exact(cpp: dict[str, Any], name: str) -> None:
    """weightSum/min/max are bit-exact.

    EXACT tier: ``min``/``max`` select a stored sample, and ``weightSum`` is
    a running sum over the same values in the same insertion order on both
    sides, so no rounding can differ.
    """
    block = cpp[name]
    stats = build(block)
    tolerance.exact(stats.weight_sum(), float(block["weight_sum"]))
    tolerance.exact(stats.min(), float(block["min"]))
    tolerance.exact(stats.max(), float(block["max"]))


@pytest.mark.parametrize("name", DATASETS)
@pytest.mark.parametrize(
    "moment",
    ["mean", "variance", "standard_deviation", "error_estimate", "skewness", "kurtosis"],
)
def test_moments_match_cpp(cpp: dict[str, Any], name: str, moment: str) -> None:
    """Empirical moments match C++ to TIGHT.

    Both sides run the same ``expectationValue`` loop in the same order; the
    only slack is the libm ``sqrt``/division rounding, which is sub-ULP.
    ``one_below`` has 4 samples, so every moment is defined; no block here
    is small enough for the arity guards to fire.
    """
    block = cpp[name]
    stats = build(block)
    tolerance.tight(getattr(stats, moment)(), float(block[moment]))


@pytest.mark.parametrize("name", DATASETS)
def test_percentile_walk_exact(cpp: dict[str, Any], name: str) -> None:
    """percentile() reproduces the C++ weight walk, bit-exact.

    EXACT tier: the result is always one of the stored samples, so agreement
    is a question of picking the same index, not of arithmetic.
    """
    block = cpp[name]
    stats = build(block)
    for y, expected in zip(block["percentile_ys"], block["percentile"], strict=True):
        tolerance.exact(stats.percentile(float(y)), float(expected))


@pytest.mark.parametrize("name", DATASETS)
def test_top_percentile_walk_exact(cpp: dict[str, Any], name: str) -> None:
    """topPercentile() reproduces the reverse C++ weight walk, bit-exact."""
    block = cpp[name]
    stats = build(block)
    for y, expected in zip(block["percentile_ys"], block["top_percentile"], strict=True):
        tolerance.exact(stats.top_percentile(float(y)), float(expected))


@pytest.mark.parametrize("name", DATASETS)
def test_expectation_value_over_range(cpp: dict[str, Any], name: str) -> None:
    """expectation_value(f, in_range) matches C++ in value and in count."""
    block = cpp[name]
    stats = build(block)
    target = float(block["target"])
    value, count = stats.expectation_value(lambda x: x * x, lambda x: x > target)
    tolerance.tight(value, float(block["expectation_value_sq_above_target"]))
    assert count == int(block["expectation_value_sq_above_target_count"])


@pytest.mark.parametrize("name", DATASETS)
def test_expectation_value_over_all(cpp: dict[str, Any], name: str) -> None:
    """The predicate-free overload averages over every sample."""
    block = cpp[name]
    stats = build(block)
    value, count = stats.expectation_value(lambda x: x * x)
    tolerance.tight(value, float(block["expectation_value_sq_all"]))
    assert count == int(block["expectation_value_sq_all_count"])


def test_expectation_value_empty_range_returns_null_sentinel() -> None:
    """An empty range yields C++'s ``(Null<Real>(), 0)``."""
    stats = GeneralStatistics()
    stats.add_sequence([1.0, 2.0, 3.0])
    value, count = stats.expectation_value(lambda x: x, lambda x: x > 100.0)
    tolerance.exact(value, NULL_REAL)
    assert count == 0


def test_percentile_out_of_range_rejected() -> None:
    """percentile is defined on (0, 1] only."""
    stats = GeneralStatistics()
    stats.add_sequence([1.0, 2.0, 3.0])
    with pytest.raises(LibraryException):
        stats.percentile(0.0)
    with pytest.raises(LibraryException):
        stats.percentile(1.5)
    with pytest.raises(LibraryException):
        stats.top_percentile(0.0)


def test_sort_orders_pairs_not_values() -> None:
    """sort() orders (value, weight) pairs, as C++'s std::sort over pairs does."""
    stats = GeneralStatistics()
    stats.add(1.0, 3.0)
    stats.add(1.0, 1.0)
    stats.add(0.5, 2.0)
    stats.sort()
    assert stats.data() == [(0.5, 2.0), (1.0, 1.0), (1.0, 3.0)]


def test_moment_arity_guards() -> None:
    """Each moment refuses to run below its minimum sample count."""
    stats = GeneralStatistics()
    with pytest.raises(LibraryException):
        stats.mean()
    stats.add(1.0)
    with pytest.raises(LibraryException):
        stats.variance()
    stats.add(2.0)
    with pytest.raises(LibraryException):
        stats.skewness()
    stats.add(3.0)
    with pytest.raises(LibraryException):
        stats.kurtosis()


def test_negative_weight_rejected() -> None:
    """Weights must be non-negative."""
    stats = GeneralStatistics()
    with pytest.raises(LibraryException):
        stats.add(1.0, weight=-1.0)
