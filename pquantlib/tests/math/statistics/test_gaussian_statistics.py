"""Cross-validate GenericGaussianStatistics / StatsHolder against the C++ probe.

Reference: ``migration-harness/references/v143/math/statistics.json`` — the
per-data-set ``gaussian_*`` keys plus the dedicated ``gaussian`` block,
which probes the same closed forms over three different underlying
statistics tools: ``GeneralStatistics``, ``IncrementalStatistics`` and
``StatsHolder``. That triple is the whole reason the C++ class is a
template, so the port's mixin has to work for all three.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.statistics.gaussian_statistics import (
    GaussianStatistics,
    GenericGaussianStatistics,
    StatsHolder,
)
from pquantlib.math.statistics.general_statistics import GeneralStatistics
from pquantlib.math.statistics.incremental_statistics import IncrementalStatistics
from pquantlib.math.statistics.statistics import Statistics
from pquantlib.testing import reference_reader, tolerance

DATASETS = ("ties", "spread", "positive", "one_below", "weighted")


class IncrementalGaussianStatistics(GenericGaussianStatistics, IncrementalStatistics):
    """# C++ parity: ``GenericGaussianStatistics<IncrementalStatistics>``."""

    __slots__ = ()


class HeldGaussianStatistics(GenericGaussianStatistics, StatsHolder):
    """# C++ parity: ``GenericGaussianStatistics<StatsHolder>``."""

    __slots__ = ()


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/statistics")


def build(block: dict[str, Any]) -> Statistics:
    """Rebuild the C++ accumulator described by a data-set block."""
    stats = Statistics()
    weights = [float(w) for w in block["weights"]]
    stats.add_sequence([float(v) for v in block["data"]], weights or None)
    return stats


def check(call: Callable[[], float], expected: Any) -> None:
    """Assert ``call()`` reproduces ``expected``, including a C++ throw."""
    if expected == "__error__":
        with pytest.raises(LibraryException):
            call()
        return
    tolerance.tight(call(), float(expected))


def test_gaussian_statistics_is_the_typedef() -> None:
    """``GaussianStatistics`` is the mixin over ``GeneralStatistics``."""
    assert issubclass(GaussianStatistics, GenericGaussianStatistics)
    assert issubclass(GaussianStatistics, GeneralStatistics)


@pytest.mark.parametrize("name", DATASETS)
def test_closed_forms_over_general_statistics(cpp: dict[str, Any], name: str) -> None:
    """Every Gaussian closed form, over each data-set block.

    TIGHT: the closed forms are a handful of multiplications around
    ``CumulativeNormalDistribution`` / ``NormalDistribution`` /
    ``InverseCumulativeNormal``, all of which are already cross-validated
    ports of the same rational approximations, so the residual is
    accumulated rounding of a dozen operations.
    """
    block = cpp[name]
    stats = build(block)
    target = float(block["target"])
    centile = float(block["centile"])
    check(stats.gaussian_downside_variance, block["gaussian_downside_variance"])
    check(stats.gaussian_downside_deviation, block["gaussian_downside_deviation"])
    check(lambda: stats.gaussian_regret(target), block["gaussian_regret"])
    check(lambda: stats.gaussian_percentile(0.75), block["gaussian_percentile"])
    check(lambda: stats.gaussian_top_percentile(0.75), block["gaussian_top_percentile"])
    check(
        lambda: stats.gaussian_potential_upside(centile),
        block["gaussian_potential_upside"],
    )
    check(lambda: stats.gaussian_value_at_risk(centile), block["gaussian_value_at_risk"])
    check(
        lambda: stats.gaussian_expected_shortfall(centile),
        block["gaussian_expected_shortfall"],
    )
    check(lambda: stats.gaussian_shortfall(target), block["gaussian_shortfall"])
    check(
        lambda: stats.gaussian_average_shortfall(target),
        block["gaussian_average_shortfall"],
    )


@pytest.mark.parametrize("name", DATASETS)
def test_gaussian_percentile_excludes_both_extremes(
    cpp: dict[str, Any], name: str
) -> None:
    """gaussianPercentile is open at 0 and at 1."""
    block = cpp[name]
    stats = build(block)
    check(lambda: stats.gaussian_percentile(0.0), block["gaussian_percentile_at_zero"])
    check(lambda: stats.gaussian_percentile(1.0), block["gaussian_percentile_at_one"])


def test_stats_holder_carries_the_moments(cpp: dict[str, Any]) -> None:
    """StatsHolder hands back exactly what it was given."""
    block = cpp["gaussian"]
    holder = StatsHolder(
        float(block["holder_mean"]), float(block["holder_standard_deviation"])
    )
    tolerance.exact(holder.mean(), float(block["holder_mean"]))
    tolerance.exact(holder.standard_deviation(), float(block["holder_standard_deviation"]))


@pytest.mark.parametrize("underlying", ["general", "incremental", "holder"])
def test_closed_forms_over_each_underlying(cpp: dict[str, Any], underlying: str) -> None:
    """The same closed forms over GeneralStatistics / Incremental / StatsHolder.

    The three tools disagree slightly on the mean and standard deviation
    they feed in (different summation schemes), and the probe records each
    one's answer separately, so this also pins that the mixin reads the
    moments from the class it was mixed into.
    """
    block = cpp["gaussian"]
    data = [float(v) for v in block["data"]]
    target = float(block["target"])
    centile = float(block["centile"])

    stats: GenericGaussianStatistics
    if underlying == "general":
        general = Statistics()
        general.add_sequence(data)
        stats = general
    elif underlying == "incremental":
        incremental = IncrementalGaussianStatistics()
        incremental.add_sequence(data)
        stats = incremental
    else:
        reference = Statistics()
        reference.add_sequence(data)
        stats = HeldGaussianStatistics(
            reference.mean(), reference.standard_deviation()
        )

    check(lambda: stats.gaussian_regret(target), block[f"{underlying}_gaussian_regret"])
    check(
        lambda: stats.gaussian_percentile(0.3),
        block[f"{underlying}_gaussian_percentile"],
    )
    check(
        lambda: stats.gaussian_potential_upside(centile),
        block[f"{underlying}_gaussian_potential_upside"],
    )
    check(
        lambda: stats.gaussian_value_at_risk(centile),
        block[f"{underlying}_gaussian_value_at_risk"],
    )
    check(
        lambda: stats.gaussian_expected_shortfall(centile),
        block[f"{underlying}_gaussian_expected_shortfall"],
    )
    check(
        lambda: stats.gaussian_shortfall(target),
        block[f"{underlying}_gaussian_shortfall"],
    )
    check(
        lambda: stats.gaussian_average_shortfall(target),
        block[f"{underlying}_gaussian_average_shortfall"],
    )


def test_incremental_underlying_downside_variance(cpp: dict[str, Any]) -> None:
    """gaussianDownsideVariance over IncrementalStatistics."""
    block = cpp["gaussian"]
    stats = IncrementalGaussianStatistics()
    stats.add_sequence([float(v) for v in block["data"]])
    check(
        stats.gaussian_downside_variance,
        block["incremental_gaussian_downside_variance"],
    )


def test_potential_upside_and_var_ranges() -> None:
    """The [0.9, 1.0) window is enforced on the Gaussian variants too."""
    stats = Statistics()
    stats.add_sequence([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(LibraryException):
        stats.gaussian_potential_upside(0.5)
    with pytest.raises(LibraryException):
        stats.gaussian_value_at_risk(1.0)
    with pytest.raises(LibraryException):
        stats.gaussian_expected_shortfall(0.89)
