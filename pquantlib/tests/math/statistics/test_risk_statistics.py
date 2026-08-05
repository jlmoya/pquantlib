"""Cross-validate GenericRiskStatistics / RiskStatistics against the C++ probe.

Reference: ``migration-harness/references/v143/math/statistics.json`` — the
five data-set blocks.

Each block records, per measure, either the C++ value or the marker
``"__error__"`` when C++ throws. Both are asserted: the ``positive`` block
has nothing below zero, so every below-target measure must raise, and the
``one_below`` block has exactly one sample below zero, which trips
``regret``'s "samples under target <= 1" guard rather than returning zero.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.statistics.gaussian_statistics import GaussianStatistics
from pquantlib.math.statistics.general_statistics import GeneralStatistics
from pquantlib.math.statistics.risk_statistics import RiskStatistics
from pquantlib.math.statistics.statistics import Statistics
from pquantlib.testing import reference_reader, tolerance

DATASETS = ("ties", "spread", "positive", "one_below", "weighted")

VOID_MEASURES = (
    "semi_variance",
    "semi_deviation",
    "downside_variance",
    "downside_deviation",
)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/statistics")


def build(block: dict[str, Any]) -> Statistics:
    """Rebuild the C++ accumulator described by a data-set block."""
    stats = Statistics()
    weights = [float(w) for w in block["weights"]]
    stats.add_sequence([float(v) for v in block["data"]], weights or None)
    return stats


def check(
    name: str, call: Callable[[], float], expected: Any, *, exact: bool = False
) -> None:
    """Assert ``call()`` reproduces ``expected``, including a C++ throw."""
    if expected == "__error__":
        with pytest.raises(LibraryException):
            call()
        return
    value = call()
    reason = f"measure={name}"
    if exact:
        tolerance.exact(value, float(expected), reason=reason)
    else:
        tolerance.tight(value, float(expected), reason=reason)


def test_statistics_is_the_risk_statistics_typedef() -> None:
    """``Statistics`` is ``RiskStatistics``, and the layering is real.

    # C++ parity: statistics.hpp ``typedef RiskStatistics Statistics``, over
    # ``GenericRiskStatistics<GaussianStatistics>`` over
    # ``GenericGaussianStatistics<GeneralStatistics>``.
    """
    assert Statistics is RiskStatistics
    assert issubclass(RiskStatistics, GaussianStatistics)
    assert issubclass(GaussianStatistics, GeneralStatistics)


@pytest.mark.parametrize("name", DATASETS)
@pytest.mark.parametrize("measure", VOID_MEASURES)
def test_void_argument_measures(cpp: dict[str, Any], name: str, measure: str) -> None:
    """semiVariance/semiDeviation/downsideVariance/downsideDeviation.

    TIGHT: each is a ``regret`` call, i.e. one ``expectationValue`` loop
    (identical order both sides) plus a scaling and possibly a ``sqrt``.
    """
    block = cpp[name]
    stats = build(block)
    check(measure, getattr(stats, measure), block[measure])


@pytest.mark.parametrize("name", DATASETS)
def test_regret_and_shortfall_at_target(cpp: dict[str, Any], name: str) -> None:
    """regret / shortfall / averageShortfall at the block's target."""
    block = cpp[name]
    stats = build(block)
    target = float(block["target"])
    check("regret", lambda: stats.regret(target), block["regret"])
    check("shortfall", lambda: stats.shortfall(target), block["shortfall"], exact=True)
    check(
        "average_shortfall",
        lambda: stats.average_shortfall(target),
        block["average_shortfall"],
    )


@pytest.mark.parametrize("name", DATASETS)
def test_percentile_driven_measures(cpp: dict[str, Any], name: str) -> None:
    """potentialUpside / valueAtRisk / expectedShortfall at the block's centile.

    The first two return a stored sample after a floor/cap, so they are
    bit-exact; expectedShortfall averages and is TIGHT.
    """
    block = cpp[name]
    stats = build(block)
    centile = float(block["centile"])
    check(
        "potential_upside",
        lambda: stats.potential_upside(centile),
        block["potential_upside"],
        exact=True,
    )
    check(
        "value_at_risk",
        lambda: stats.value_at_risk(centile),
        block["value_at_risk"],
        exact=True,
    )
    check(
        "expected_shortfall",
        lambda: stats.expected_shortfall(centile),
        block["expected_shortfall"],
    )


@pytest.mark.parametrize("name", DATASETS)
def test_centile_range_guards(cpp: dict[str, Any], name: str) -> None:
    """The [0.9, 1.0) window is enforced at both ends."""
    block = cpp[name]
    stats = build(block)
    check(
        "potential_upside_below_range",
        lambda: stats.potential_upside(0.5),
        block["potential_upside_below_range"],
    )
    check(
        "value_at_risk_at_one",
        lambda: stats.value_at_risk(1.0),
        block["value_at_risk_at_one"],
    )


def test_below_target_is_strict() -> None:
    """A sample sitting exactly on the target is not "below" it.

    Two samples at the target plus two strictly below: ``shortfall`` counts
    2 of 4, and ``regret`` averages only the two strictly-below ones.
    """
    stats = Statistics()
    stats.add_sequence([0.0, 0.0, -1.0, -3.0])
    tolerance.exact(stats.shortfall(0.0), 0.5)
    # regret(0) = N/(N-1) * mean(x^2 | x<0) = 2/1 * (1+9)/2 = 10
    tolerance.tight(stats.regret(0.0), 10.0)


def test_regret_needs_more_than_one_sample_below_target() -> None:
    """One sample below the target is an error, not a degenerate zero."""
    stats = Statistics()
    stats.add_sequence([5.0, 6.0, -1.0])
    with pytest.raises(LibraryException):
        stats.regret(0.0)


def test_average_shortfall_needs_data_below_target() -> None:
    """No sample below the target is an error, not a zero."""
    stats = Statistics()
    stats.add_sequence([5.0, 6.0, 7.0])
    with pytest.raises(LibraryException):
        stats.average_shortfall(0.0)


def test_expected_shortfall_averages_below_negated_var() -> None:
    """expectedShortfall's target is ``-valueAtRisk``, not ``percentile(1-p)``.

    With every sample positive, ``valueAtRisk`` floors to 0.0, so the
    averaging range is ``x < 0`` — which is empty, and C++ raises.
    """
    stats = Statistics()
    stats.add_sequence([1.0, 2.0, 3.0, 4.0, 5.0])
    # ``-min(1.0, 0.0)`` is a *negative* zero; the sign is what tells
    # ``-min(x, 0)`` apart from ``max(-x, 0)``.
    tolerance.exact(stats.value_at_risk(0.95), -0.0)
    with pytest.raises(LibraryException):
        stats.expected_shortfall(0.95)


def test_shortfall_on_empty_set_rejected() -> None:
    """shortfall requires at least one sample."""
    stats = Statistics()
    with pytest.raises(LibraryException):
        stats.shortfall(0.0)
