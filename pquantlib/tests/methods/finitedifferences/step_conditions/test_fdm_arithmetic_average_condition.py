"""Cross-validation tests for ``FdmArithmeticAverageCondition``.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmarithmeticaveragecondition.{hpp,cpp}
# @ v1.43.

Reference: ``migration-harness/references/v143/methods/stepconditions.json``,
``avg_*`` keys.

The condition delegates to ``MonotonicCubicNaturalSpline`` **with
extrapolation enabled**. The fixture puts the equity grid ([50, 150]) outside
the average grid ([60, 140]) precisely so that the interpolant is queried
below and above its own data range at the extreme equity nodes: a clamping
interpolant would return the edge values there, whereas C++ continues the
edge cubic and produces values far outside the data range (the ``t = 0.25``
case lands at about -13.6 against grid data in [0.25, 19]). That divergence
is what these cases exist to catch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.step_conditions.fdm_arithmetic_average_condition import (
    FdmArithmeticAverageCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)
from tests.methods.finitedifferences.step_conditions import _fixtures

_AVERAGE_TIMES = [0.25, 0.5, 0.75]


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return _fixtures.load()


def _condition(
    past_fixings: int = 0,
    equity_direction: int = 0,
    average_times: list[float] | None = None,
) -> FdmArithmeticAverageCondition:
    return FdmArithmeticAverageCondition(
        _AVERAGE_TIMES if average_times is None else average_times,
        0.0,
        past_fixings,
        _fixtures.average_mesher(),
        equity_direction,
    )


def test_is_a_step_condition() -> None:
    assert isinstance(_condition(), StepCondition)


def test_mesher_layout_matches_probe(reference: dict[str, Any]) -> None:
    mesher = _fixtures.average_mesher()
    assert mesher.layout().size() == reference["avg_size"]
    assert list(mesher.layout().dim()) == reference["avg_dim"]
    _fixtures.assert_array_tight(mesher.locations(0), reference, "avg_locations_0")
    _fixtures.assert_array_tight(mesher.locations(1), reference, "avg_locations_1")


@pytest.mark.parametrize(
    ("equity_direction", "x_key", "a_key"),
    [(0, "avg_dir0_x", "avg_dir0_a"), (1, "avg_dir1_x", "avg_dir1_a")],
)
def test_physical_grids_derived_in_constructor(
    reference: dict[str, Any], equity_direction: int, x_key: str, a_key: str
) -> None:
    """Pins the constructor's grid-derivation recipe against C++.

    ``x_``/``a_`` are private, so rather than reaching into the instance this
    replays the exact recipe the constructor uses — stride the flat locations
    array by ``spacing[direction]`` and exponentiate — and compares it to the
    values the probe read out of the C++ object. The derived grids then feed
    every ``apply_to`` comparison in this module, which is where a mismatch
    would actually bite.
    """
    mesher = _fixtures.average_mesher()
    layout = mesher.layout()
    average_direction = 1 if equity_direction == 0 else 0

    x_spacing = layout.spacing()[equity_direction]
    locs = mesher.locations(equity_direction)
    xs = np.array(
        [np.exp(float(locs[i * x_spacing])) for i in range(layout.dim()[equity_direction])],
        dtype=np.float64,
    )
    a_spacing = layout.spacing()[average_direction]
    locs = mesher.locations(average_direction)
    avgs = np.array(
        [np.exp(float(locs[i * a_spacing])) for i in range(layout.dim()[average_direction])],
        dtype=np.float64,
    )
    _fixtures.assert_array_tight(xs, reference, x_key)
    _fixtures.assert_array_tight(avgs, reference, a_key)


def test_no_op_away_from_a_fixing(reference: dict[str, Any]) -> None:
    a: Array = _fixtures.ref_array(reference, "avg_seed")
    _condition().apply_to(a, 0.1)
    _fixtures.assert_array_tight(a, reference, "avg_dir0_pf0_no_fixing")


@pytest.mark.parametrize(
    ("t", "key"),
    [
        # iT = 1, nTimes = 1 -> weights (0, 1): pure equity, deepest extrapolation.
        (0.25, "avg_dir0_pf0_t025"),
        # iT = 2 -> weights (1/2, 1/2).
        (0.5, "avg_dir0_pf0_t050"),
        # iT = 3 -> weights (2/3, 1/3).
        (0.75, "avg_dir0_pf0_t075"),
    ],
)
def test_average_update_no_past_fixings(reference: dict[str, Any], t: float, key: str) -> None:
    a: Array = _fixtures.ref_array(reference, "avg_seed")
    _condition().apply_to(a, t)
    _fixtures.assert_array_tight(a, reference, key)


@pytest.mark.parametrize(
    ("t", "key"),
    [(0.25, "avg_dir0_pf2_t025"), (0.75, "avg_dir0_pf2_t075")],
)
def test_past_fixings_shift_the_weights(reference: dict[str, Any], t: float, key: str) -> None:
    """``iT = index + 1 + pastFixings``: two past fixings move every weight."""
    a: Array = _fixtures.ref_array(reference, "avg_seed")
    _condition(past_fixings=2).apply_to(a, t)
    _fixtures.assert_array_tight(a, reference, key)


def test_equity_direction_one(reference: dict[str, Any]) -> None:
    """With ``equityDirection = 1`` the roles of the two axes swap."""
    a: Array = _fixtures.ref_array(reference, "avg_seed")
    _condition(equity_direction=1).apply_to(a, 0.5)
    _fixtures.assert_array_tight(a, reference, "avg_dir1_pf0_t050")


def test_repeated_fixing_time(reference: dict[str, Any]) -> None:
    """Two fixings on the same date: ``nTimes = 2`` while ``iter`` stays on the first.

    ``iT`` is therefore 2 and the weights collapse to ``(0, 1)`` — the same
    query point as the single-fixing ``t = 0.25`` case, which the reference
    confirms.
    """
    a: Array = _fixtures.ref_array(reference, "avg_seed")
    _condition(average_times=list(reference["avg_dup_times"])).apply_to(a, 0.5)
    _fixtures.assert_array_tight(a, reference, "avg_dir0_dup_t050")


def test_extrapolation_leaves_the_data_range(reference: dict[str, Any]) -> None:
    """Guard on the guard: the pinned case really does extrapolate.

    If a future change made the interpolant clamp, this assertion would still
    hold on the reference but the value comparison above would fail — so this
    test documents *why* the fixture is shaped the way it is, by showing the
    reference output escapes the convex hull of the seed values.
    """
    seed: list[float] = reference["avg_seed"]
    out: list[float] = reference["avg_dir0_pf0_t025"]
    assert min(out) < min(seed)


def test_requires_two_dimensions() -> None:
    """C++ ``QL_REQUIRE(mesher->layout()->dim().size() == 2, "2D allowed only")``."""
    mesher = FdmMesherComposite(Uniform1dMesher(0.0, 1.0, 5))
    with pytest.raises(LibraryException, match="2D allowed only"):
        FdmArithmeticAverageCondition(_AVERAGE_TIMES, 0.0, 0, mesher, 0)


def test_requires_equity_direction_zero_or_one() -> None:
    """C++ ``QL_REQUIRE(equityDirection == 0 || equityDirection == 1, ...)``."""
    with pytest.raises(LibraryException, match="equityDirection"):
        FdmArithmeticAverageCondition(_AVERAGE_TIMES, 0.0, 0, _fixtures.average_mesher(), 2)


def test_inconsistent_array_dimensions_raises() -> None:
    """C++ asserts at the top of ``applyTo``, before the fixing-time test."""
    bad: Array = np.zeros(3, dtype=np.float64)
    with pytest.raises(LibraryException):
        _condition().apply_to(bad, 0.1)


def test_running_accumulator_argument_is_ignored(reference: dict[str, Any]) -> None:
    """The second constructor argument is unnamed in C++ and never stored."""
    mesher = _fixtures.average_mesher()
    a: Array = _fixtures.ref_array(reference, "avg_seed")
    FdmArithmeticAverageCondition(_AVERAGE_TIMES, 1234.5, 0, mesher, 0).apply_to(a, 0.5)
    _fixtures.assert_array_tight(a, reference, "avg_dir0_pf0_t050")
