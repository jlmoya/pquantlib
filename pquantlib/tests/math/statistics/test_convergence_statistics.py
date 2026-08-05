"""Cross-validate ConvergenceStatistics / DoublingConvergenceSteps.

Reference: ``migration-harness/references/v143/math/statistics.json`` —
``convergence``. The whole table is pinned, for both underlying statistics
types, in three situations: a plain 8-sample run, a run after ``reset()``
(which must clear the table *and* rewind the sampling rule), and a weighted
``add_sequence`` run, which only records the right rows if the sequence
helper routes through the overriding ``add``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.math.statistics.convergence_statistics import (
    ConvergenceStatistics,
    DoublingConvergenceSteps,
)
from pquantlib.math.statistics.incremental_statistics import IncrementalStatistics
from pquantlib.math.statistics.statistics import Statistics
from pquantlib.testing import reference_reader, tolerance


class ConvergingStatistics(ConvergenceStatistics, Statistics):
    """# C++ parity: ``ConvergenceStatistics<Statistics>``."""


class ConvergingIncrementalStatistics(ConvergenceStatistics, IncrementalStatistics):
    """# C++ parity: ``ConvergenceStatistics<IncrementalStatistics>``."""


UNDERLYINGS = {
    "statistics": ConvergingStatistics,
    "incremental": ConvergingIncrementalStatistics,
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/statistics")


def test_doubling_rule_step_sequence(cpp: dict[str, Any]) -> None:
    """The doubling policy emits 1, 3, 7, 15, … exactly."""
    expected = [int(n) for n in cpp["convergence"]["doubling_steps"]]
    rule = DoublingConvergenceSteps()
    steps: list[int] = []
    current = rule.initial_samples()
    for _ in range(len(expected)):
        steps.append(current)
        current = rule.next_samples(current)
    assert steps == expected


@pytest.mark.parametrize("name", list(UNDERLYINGS))
def test_convergence_table(cpp: dict[str, Any], name: str) -> None:
    """The full table after eight unit-weight samples.

    Sample counts are integers and compared exactly; the recorded means go
    through the underlying tool's own mean, hence TIGHT.
    """
    block = cpp["convergence"][name]
    stats = UNDERLYINGS[name]()
    for i in range(1, 9):
        stats.add(float(i))
    table = stats.convergence_table()
    assert [row[0] for row in table] == [int(n) for n in block["first_sizes"]]
    for row, expected in zip(table, block["first_means"], strict=True):
        tolerance.tight(row[1], float(expected))


@pytest.mark.parametrize("name", list(UNDERLYINGS))
def test_reset_rewinds_table_and_rule(cpp: dict[str, Any], name: str) -> None:
    """reset() clears the table and restarts the sampling rule at 1."""
    block = cpp["convergence"][name]
    stats = UNDERLYINGS[name]()
    for i in range(1, 9):
        stats.add(float(i))
    stats.reset()
    assert stats.convergence_table() == []
    for i in range(1, 5):
        stats.add(float(i))
    table = stats.convergence_table()
    assert [row[0] for row in table] == [int(n) for n in block["after_reset_sizes"]]
    for row, expected in zip(table, block["after_reset_means"], strict=True):
        tolerance.tight(row[1], float(expected))


@pytest.mark.parametrize("name", list(UNDERLYINGS))
def test_weighted_add_sequence_records_steps(cpp: dict[str, Any], name: str) -> None:
    """add_sequence with weights must still record at every sampling step."""
    block = cpp["convergence"][name]
    stats = UNDERLYINGS[name]()
    stats.add_sequence(
        [float(v) for v in block["weighted_values"]],
        [float(w) for w in block["weighted_weights"]],
    )
    table = stats.convergence_table()
    assert [row[0] for row in table] == [int(n) for n in block["weighted_sizes"]]
    for row, expected in zip(table, block["weighted_means"], strict=True):
        tolerance.tight(row[1], float(expected))


def test_custom_sampling_rule_is_honoured() -> None:
    """A non-default policy object replaces the doubling rule.

    # C++ parity: the ``U`` template parameter, defaulted to
    # ``DoublingConvergenceSteps``.
    """

    class EveryThird:
        def initial_samples(self) -> int:
            return 3

        def next_samples(self, current: int) -> int:
            return current + 3

    stats = ConvergingStatistics(sampling_rule=EveryThird())
    for i in range(1, 10):
        stats.add(float(i))
    assert [row[0] for row in stats.convergence_table()] == [3, 6, 9]


def test_decorated_statistics_keeps_the_underlying_surface() -> None:
    """The decorator is transparent: the underlying inspectors still work."""
    stats = ConvergingStatistics()
    stats.add_sequence([1.0, 2.0, 3.0, 4.0])
    assert stats.samples() == 4
    tolerance.exact(stats.mean(), 2.5)
    tolerance.exact(stats.percentile(0.5), 2.0)
