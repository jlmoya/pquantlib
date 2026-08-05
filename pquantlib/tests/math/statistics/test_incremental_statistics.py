"""Cross-validate IncrementalStatistics against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/statistics.json`` —
``incremental``.

The C++ class is a facade over four *different* boost accumulators, and
this file pins that the port transcribed each of them rather than picking
one online scheme:

* ``mean()`` is boost's **lazy** mean, ``weighted_sum / sum_of_weights``;
* ``variance()`` is boost's **immediate** (iterative) variance, which runs
  off a separately maintained ``immediate_weighted_mean`` — the
  ``shifted`` data set (samples ~1e8 with a variance of ~0.01) is where a
  naive ``m2 - m1^2`` loses every digit and this recurrence does not;
* ``skewness()``/``kurtosis()`` are boost's **lazy** estimators in raw
  weighted moments;
* a zero-weight sample still increments the sample count.

The ``shifted`` set pins mean and variance only. Its skewness and kurtosis
go through the lazy estimators, whose numerator cancels from ~1e8 down to
an exact value of -1.63e-2 — zero surviving significant digits — so no
reference value for them is reproducible in double precision and none is
recorded.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.statistics.incremental_statistics import IncrementalStatistics
from pquantlib.testing import reference_reader, tolerance

SCALARS = (
    "weight_sum",
    "mean",
    "variance",
    "standard_deviation",
    "error_estimate",
    "skewness",
    "kurtosis",
    "min",
    "max",
    "downside_weight_sum",
    "downside_variance",
    "downside_deviation",
)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/statistics")


def build(cpp: dict[str, Any]) -> IncrementalStatistics:
    """Rebuild the probe's weighted accumulator."""
    block = cpp["incremental"]
    stats = IncrementalStatistics()
    stats.add_sequence(
        [float(v) for v in block["values"]], [float(w) for w in block["weights"]]
    )
    return stats


def test_sample_counts(cpp: dict[str, Any]) -> None:
    """Total and downside sample counts."""
    block = cpp["incremental"]
    stats = build(cpp)
    assert stats.samples() == int(block["samples"])
    assert stats.downside_samples() == int(block["downside_samples"])


@pytest.mark.parametrize("scalar", SCALARS)
def test_weighted_accumulators(cpp: dict[str, Any], scalar: str) -> None:
    """Each boost accumulator's own formula, on weighted data.

    TIGHT: the port accumulates the same running sums in the same order,
    so only the final divisions and roots can differ, sub-ULP. The data
    set has a single negative sample, so the two downside inspectors must
    raise, exactly as C++ does — that is recorded as ``"__error__"``.
    """
    block = cpp["incremental"]
    stats = build(cpp)
    expected = block[scalar]
    if expected == "__error__":
        with pytest.raises(LibraryException):
            getattr(stats, scalar)()
        return
    tolerance.tight(getattr(stats, scalar)(), float(expected))


def test_shifted_mean_and_variance(cpp: dict[str, Any]) -> None:
    """The numerically hostile set: 64 weighted samples around 1e8.

    This is the case the C++ test-suite calls out as one the pre-boost
    implementation failed. TIGHT holds only because the port uses the
    lazy mean for ``mean()`` and the iterative recurrence for
    ``variance()`` — swapping either loses digits here.
    """
    block = cpp["incremental"]
    stats = IncrementalStatistics()
    stats.add_sequence(
        [float(v) for v in block["shifted_values"]],
        [float(w) for w in block["shifted_weights"]],
    )
    tolerance.tight(stats.mean(), float(block["shifted_mean"]))
    tolerance.tight(stats.variance(), float(block["shifted_variance"]))


def test_zero_weight_sample_still_counts(cpp: dict[str, Any]) -> None:
    """A zero-weight sample increments the count and updates min/max."""
    block = cpp["incremental"]
    stats = IncrementalStatistics()
    stats.add(1.0, 1.0)
    stats.add(9.0, 0.0)
    stats.add(3.0, 1.0)
    assert stats.samples() == int(block["zero_weight_samples"])
    tolerance.exact(stats.weight_sum(), float(block["zero_weight_weight_sum"]))
    tolerance.tight(stats.mean(), float(block["zero_weight_mean"]))
    tolerance.exact(stats.max(), float(block["zero_weight_max"]))


def test_reset_clears_every_accumulator(cpp: dict[str, Any]) -> None:
    """reset() rewinds all four accumulators, not just the count."""
    stats = build(cpp)
    stats.reset()
    assert stats.samples() == 0
    assert stats.downside_samples() == 0
    tolerance.exact(stats.weight_sum(), 0.0)
    with pytest.raises(LibraryException):
        stats.mean()


def test_arity_guards() -> None:
    """Each inspector refuses to run below its minimum sample count."""
    stats = IncrementalStatistics()
    with pytest.raises(LibraryException):
        stats.mean()
    with pytest.raises(LibraryException):
        stats.min()
    stats.add(1.0)
    with pytest.raises(LibraryException):
        stats.variance()
    stats.add(2.0)
    with pytest.raises(LibraryException):
        stats.skewness()
    stats.add(3.0)
    with pytest.raises(LibraryException):
        stats.kurtosis()


def test_downside_needs_more_than_one_negative() -> None:
    """One negative sample is not enough for the downside variance."""
    stats = IncrementalStatistics()
    stats.add_sequence([1.0, 2.0, -1.0])
    with pytest.raises(LibraryException):
        stats.downside_variance()


def test_negative_weight_rejected() -> None:
    """Weights must be non-negative."""
    stats = IncrementalStatistics()
    with pytest.raises(LibraryException):
        stats.add(1.0, weight=-1.0)
