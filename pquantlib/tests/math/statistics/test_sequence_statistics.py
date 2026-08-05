"""Cross-validate GenericSequenceStatistics and its two instantiations.

Reference: ``migration-harness/references/v143/math/statistics.json`` —
``sequence``. Eight 3-dimensional samples with non-uniform weights, run
through both C++ element types (``Statistics`` and
``IncrementalStatistics``), plus the risk surface that only the
``Statistics`` element supports.

The data deliberately carries at least two negatives per dimension, so
``downsideVariance`` and ``semiVariance`` are defined rather than raising.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.statistics.incremental_statistics import IncrementalStatistics
from pquantlib.math.statistics.sequence_statistics import (
    GenericSequenceStatistics,
    SequenceStatistics,
    SequenceStatisticsInc,
)
from pquantlib.math.statistics.statistics import Statistics
from pquantlib.testing import reference_reader, tolerance

ELEMENTS = {"statistics": SequenceStatistics, "incremental": SequenceStatisticsInc}

SHARED_VECTORS = (
    "mean",
    "variance",
    "standard_deviation",
    "error_estimate",
    "skewness",
    "kurtosis",
    "min",
    "max",
    "downside_variance",
    "downside_deviation",
)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/statistics")


def build(
    cpp: dict[str, Any], cls: type[SequenceStatistics] | type[SequenceStatisticsInc]
) -> Any:
    """Rebuild the probe's accumulator for the given element type."""
    block = cpp["sequence"]
    stats = cls(3)
    for sample, weight in zip(block["samples_input"], block["weights"], strict=True):
        stats.add([float(c) for c in sample], float(weight))
    return stats


def test_typedef_element_types() -> None:
    """Both typedefs are real subclasses carrying their element type."""
    assert issubclass(SequenceStatistics, GenericSequenceStatistics)
    assert issubclass(SequenceStatisticsInc, GenericSequenceStatistics)
    assert SequenceStatistics.statistics_type is Statistics
    assert SequenceStatisticsInc.statistics_type is IncrementalStatistics


@pytest.mark.parametrize("name", list(ELEMENTS))
def test_size_samples_weight_sum(cpp: dict[str, Any], name: str) -> None:
    """Dimension, sample count and weight sum are taken from component 0."""
    block = cpp["sequence"][name]
    stats = build(cpp, ELEMENTS[name])
    assert stats.size() == int(block["size"])
    assert stats.samples() == int(block["samples"])
    tolerance.exact(stats.weight_sum(), float(block["weight_sum"]))


@pytest.mark.parametrize("name", list(ELEMENTS))
@pytest.mark.parametrize("vector", SHARED_VECTORS)
def test_shared_lifted_vectors(cpp: dict[str, Any], name: str, vector: str) -> None:
    """Every lifted per-dimension vector both element types share.

    TIGHT: each entry is the underlying 1-D inspector, already validated
    against C++ at that tier.
    """
    block = cpp["sequence"][name]
    stats = build(cpp, ELEMENTS[name])
    got = getattr(stats, vector)()
    expected = block[vector]
    assert len(got) == len(expected)
    for a, e in zip(got, expected, strict=True):
        tolerance.tight(a, float(e))


@pytest.mark.parametrize("name", list(ELEMENTS))
@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_matrices(cpp: dict[str, Any], name: str, matrix: str) -> None:
    """Covariance and correlation match C++ to TIGHT, entry by entry."""
    block = cpp["sequence"][name]
    stats = build(cpp, ELEMENTS[name])
    got = getattr(stats, matrix)()
    expected = block[matrix]
    assert got.shape == (len(expected), len(expected[0]))
    for i, row in enumerate(expected):
        for j, want in enumerate(row):
            tolerance.tight(float(got[i, j]), float(want))


def test_correlation_diagonal_is_unit(cpp: dict[str, Any]) -> None:
    """The rescale leaves an exact unit diagonal — the aliasing trap."""
    stats = build(cpp, SequenceStatistics)
    correlation = stats.correlation()
    for i in range(3):
        tolerance.tight(float(correlation[i, i]), 1.0)


def test_risk_surface_lifted(cpp: dict[str, Any]) -> None:
    """The single-argument lifted methods, available on SequenceStatistics.

    # C++ parity: the DEFINE_SEQUENCE_STAT_CONST_METHOD_DOUBLE block. C++
    # only instantiates these for an element type that has them; Python
    # declares them on the ``Statistics``-element subclass for the same
    # reason.
    """
    block = cpp["sequence"]["statistics_risk"]
    stats = build(cpp, SequenceStatistics)
    target = float(block["target"])
    centile = float(block["centile"])

    cases: dict[str, Callable[[], list[float]]] = {
        "semi_variance": stats.semi_variance,
        "semi_deviation": stats.semi_deviation,
        "percentile": lambda: stats.percentile(0.5),
        "gaussian_percentile": lambda: stats.gaussian_percentile(0.5),
        "potential_upside": lambda: stats.potential_upside(centile),
        "gaussian_potential_upside": lambda: stats.gaussian_potential_upside(centile),
        "value_at_risk": lambda: stats.value_at_risk(centile),
        "gaussian_value_at_risk": lambda: stats.gaussian_value_at_risk(centile),
        "expected_shortfall": lambda: stats.expected_shortfall(centile),
        "gaussian_expected_shortfall": lambda: stats.gaussian_expected_shortfall(centile),
        "regret": lambda: stats.regret(target),
        "shortfall": lambda: stats.shortfall(target),
        "gaussian_shortfall": lambda: stats.gaussian_shortfall(target),
        "average_shortfall": lambda: stats.average_shortfall(target),
        "gaussian_average_shortfall": lambda: stats.gaussian_average_shortfall(target),
    }
    for name, call in cases.items():
        expected = block[name]
        # A lifted method fails as soon as *one* component fails; C++ records
        # that as "__error__" for the whole vector. Here it is
        # ``expected_shortfall``: no component has a sample below its own
        # negated VaR.
        if expected == "__error__":
            with pytest.raises(LibraryException):
                call()
            continue
        got = call()
        assert len(got) == len(expected), name
        for a, e in zip(got, expected, strict=True):
            tolerance.tight(a, float(e))


def test_auto_sizes_on_first_add() -> None:
    """A default-constructed instance takes its dimension from the first add."""
    stats = SequenceStatistics()
    assert stats.size() == 0
    stats.add([1.0, 2.0])
    assert stats.size() == 2


def test_sample_size_mismatch_rejected() -> None:
    """Later samples must match the established dimension."""
    stats = SequenceStatistics(2)
    with pytest.raises(LibraryException):
        stats.add([1.0, 2.0, 3.0])


def test_covariance_needs_more_than_one_sample() -> None:
    """One sample is not enough for an unbiased covariance."""
    stats = SequenceStatistics(2)
    stats.add([1.0, 2.0])
    with pytest.raises(LibraryException):
        stats.covariance()


def test_reset_clears_the_quadratic_sum() -> None:
    """reset() on the same dimension zeroes the accumulated outer products."""
    stats = SequenceStatistics(2)
    stats.add([1.0, 2.0])
    stats.add([3.0, 4.0])
    stats.reset(2)
    assert stats.samples() == 0
    stats.add([1.0, 2.0])
    stats.add([3.0, 4.0])
    covariance = stats.covariance()
    # (1,3) and (2,4): variance 2 in both components, covariance 2.
    assert np.allclose(covariance, np.array([[2.0, 2.0], [2.0, 2.0]]))
