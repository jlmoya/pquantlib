"""Cross-validation tests for ``FdmSimpleSwingCondition``.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmsimpleswingcondition.{hpp,cpp}
# @ v1.43.

Reference: ``migration-harness/references/v143/methods/stepconditions.json``,
``swing_*`` keys.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.step_conditions.fdm_simple_swing_condition import (
    FdmSimpleSwingCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)
from tests.methods.finitedifferences.step_conditions import _fixtures

_EXERCISE_TIMES = [0.25, 0.5, 0.75]


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return _fixtures.load()


def _condition(swing_direction: int = 1, min_exercises: int = 0) -> FdmSimpleSwingCondition:
    mesher = _fixtures.swing_mesher()
    return FdmSimpleSwingCondition(
        _EXERCISE_TIMES,
        mesher,
        _fixtures.call_100(mesher),
        swing_direction,
        min_exercises,
    )


def test_is_a_step_condition() -> None:
    assert isinstance(_condition(), StepCondition)


def test_mesher_layout_matches_probe(reference: dict[str, Any]) -> None:
    mesher = _fixtures.swing_mesher()
    assert mesher.layout().size() == reference["swing_size"]
    assert list(mesher.layout().dim()) == reference["swing_dim"]
    _fixtures.assert_array_tight(mesher.locations(0), reference, "swing_locations_0")
    _fixtures.assert_array_tight(mesher.locations(1), reference, "swing_locations_1")


def test_inner_value_stub_matches_cpp(reference: dict[str, Any]) -> None:
    mesher = _fixtures.swing_mesher()
    values = _fixtures.inner_values(mesher, _fixtures.call_100(mesher), 0.5)
    _fixtures.assert_array_tight(values, reference, "swing_inner")


def test_no_op_away_from_an_exercise_time(reference: dict[str, Any]) -> None:
    a: Array = _fixtures.ref_array(reference, "swing_seed")
    _condition().apply_to(a, 0.1)
    _fixtures.assert_array_tight(a, reference, "swing_min0_no_exercise")


@pytest.mark.parametrize(
    ("t", "key"),
    [
        (0.25, "swing_min0_t025"),
        (0.5, "swing_min0_t050"),
        (0.75, "swing_min0_t075"),
    ],
)
def test_exercise_decision_without_minimum(reference: dict[str, Any], t: float, key: str) -> None:
    """No minimum: exercise only when it strictly improves the value."""
    a: Array = _fixtures.ref_array(reference, "swing_seed")
    _condition().apply_to(a, t)
    _fixtures.assert_array_tight(a, reference, key)


@pytest.mark.parametrize(
    ("min_exercises", "t", "key"),
    [
        (2, 0.5, "swing_min2_t050"),
        (2, 0.75, "swing_min2_t075"),
        (3, 0.25, "swing_min3_t025"),
    ],
)
def test_minimum_exercises_forces_exercise(
    reference: dict[str, Any], min_exercises: int, t: float, key: str
) -> None:
    """``exercisesUsed + d <= minExercises`` forces exercise regardless of value.

    ``d`` is the count of exercise times from the current one to the end, so
    the constraint bites harder the later the sweep is.
    """
    a: Array = _fixtures.ref_array(reference, "swing_seed")
    _condition(min_exercises=min_exercises).apply_to(a, t)
    _fixtures.assert_array_tight(a, reference, key)


def test_swing_direction_zero(reference: dict[str, Any]) -> None:
    """``swing_direction`` really selects the axis (here the log-spot one)."""
    a: Array = _fixtures.ref_array(reference, "swing_seed")
    _condition(swing_direction=0).apply_to(a, 0.5)
    _fixtures.assert_array_tight(a, reference, "swing_dir0_t050")


def test_min_exercises_defaults_to_zero(reference: dict[str, Any]) -> None:
    mesher = _fixtures.swing_mesher()
    cond = FdmSimpleSwingCondition(_EXERCISE_TIMES, mesher, _fixtures.call_100(mesher), 1)
    a: Array = _fixtures.ref_array(reference, "swing_seed")
    cond.apply_to(a, 0.5)
    _fixtures.assert_array_tight(a, reference, "swing_default_min_t050")


def test_absorbing_state_is_never_updated(reference: dict[str, Any]) -> None:
    """Nodes at ``coor[swing_direction] == dim-1`` have no rights left.

    C++ skips them entirely (``exercisesUsed < maxExerciseValue``), so they
    must come out of ``applyTo`` exactly as they went in.
    """
    mesher = _fixtures.swing_mesher()
    seed: Array = _fixtures.ref_array(reference, "swing_seed")
    a = seed.copy()
    _condition(min_exercises=3).apply_to(a, 0.25)
    layout = mesher.layout()
    last = layout.dim()[1] - 1
    for iterator in layout.iter():
        if iterator.coordinates[1] == last:
            assert float(a[iterator.index]) == float(seed[iterator.index])


def test_apply_to_mutates_in_place(reference: dict[str, Any]) -> None:
    """C++ ``a = retVal`` writes back into the caller's array."""
    a: Array = _fixtures.ref_array(reference, "swing_seed")
    original = a
    _condition().apply_to(a, 0.5)
    assert a is original
    _fixtures.assert_array_tight(a, reference, "swing_min0_t050")


def test_inconsistent_array_dimensions_raises() -> None:
    bad: Array = np.zeros(3, dtype=np.float64)
    with pytest.raises(LibraryException):
        _condition().apply_to(bad, 0.5)
