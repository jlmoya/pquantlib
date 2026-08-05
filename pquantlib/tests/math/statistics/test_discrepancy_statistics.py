"""Cross-validate DiscrepancyStatistics against the C++ probe.

Reference: ``migration-harness/references/v143/math/statistics.json`` —
``discrepancy``, for dimensions 2, 3 and 5.

The reference records the discrepancy after **every** ``add``, not only at
the end. That is deliberate: the C++ class folds each new point into two
running sums, and a batch recomputation of the L2 discrepancy would agree
on the final value while disagreeing on every intermediate one — and it is
the intermediates that the low-discrepancy test-suite samples as the
sequence grows.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.statistics.discrepancy_statistics import DiscrepancyStatistics
from pquantlib.testing import reference_reader, tolerance

CASES = ("dim2", "dim3", "dim5")


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/statistics")


@pytest.mark.parametrize("case", CASES)
def test_running_discrepancy(cpp: dict[str, Any], case: str) -> None:
    """Discrepancy after each add matches C++ to TIGHT.

    TIGHT rather than EXACT: ``discrepancy()`` combines three running sums
    of products and takes a square root, so the order in which the products
    are formed leaves a sub-ULP residual that the sqrt propagates.
    """
    block = cpp["discrepancy"][case]
    stats = DiscrepancyStatistics(int(block["dimension"]))
    points = [[float(c) for c in point] for point in block["points"]]
    expected = [float(d) for d in block["running_discrepancy"]]

    running: list[float] = []
    for point in points:
        stats.add(point)
        running.append(stats.discrepancy())

    assert stats.samples() == int(block["samples"])
    for got, want in zip(running, expected, strict=True):
        tolerance.tight(got, want)


@pytest.mark.parametrize("case", CASES)
def test_discrepancy_decreases_over_the_sequence(cpp: dict[str, Any], case: str) -> None:
    """Sanity: the pinned sequences do converge, so the walk is meaningful."""
    block = cpp["discrepancy"][case]
    values = [float(d) for d in block["running_discrepancy"]]
    assert values[-1] < values[0]


def test_dimension_one_rejected() -> None:
    """# C++ parity: ``QL_REQUIRE(dimension != 1, "dimension==1 not allowed")``."""
    with pytest.raises(LibraryException):
        DiscrepancyStatistics(1)


def test_reset_keeps_the_current_dimension(cpp: dict[str, Any]) -> None:
    """reset(0) keeps the dimension and rewinds the running sums.

    Replaying the same points after a reset must reproduce the same walk.
    """
    block = cpp["discrepancy"]["dim2"]
    points = [[float(c) for c in point] for point in block["points"]]
    expected = [float(d) for d in block["running_discrepancy"]]

    stats = DiscrepancyStatistics(2)
    for point in points:
        stats.add(point)
    stats.reset()
    assert stats.samples() == 0
    assert stats.size() == 2

    for point, want in zip(points, expected, strict=True):
        stats.add(point)
        tolerance.tight(stats.discrepancy(), want)


def test_is_a_sequence_statistics(cpp: dict[str, Any]) -> None:
    """The class also carries the full sequence-statistics surface.

    # C++ parity: ``class DiscrepancyStatistics : public SequenceStatistics``.
    """
    block = cpp["discrepancy"]["dim2"]
    stats = DiscrepancyStatistics(2)
    for point in block["points"]:
        stats.add([float(c) for c in point])
    assert len(stats.mean()) == 2
    assert stats.covariance().shape == (2, 2)
