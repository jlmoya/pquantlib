"""Cross-validation tests for ``FdmSimpleStorageCondition``.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmsimplestoragecondition.{hpp,cpp}
# @ v1.43.

Reference: ``migration-harness/references/v143/methods/stepconditions.json``,
``storage_*`` keys.

The condition delegates to ``BilinearInterpolation``. The reference cases are
chosen so that delegation is exercised off-grid in both volume directions and
across the three distinct regimes of the "scan the intermediate grid volumes"
loop (change rate below / equal to / above the grid step), because a
clamping-instead-of-interpolating 2-D interpolant would only be caught there.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.step_conditions.fdm_simple_storage_condition import (
    FdmSimpleStorageCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)
from tests.methods.finitedifferences.step_conditions import _fixtures

_EXERCISE_TIMES = [0.25, 0.5]


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return _fixtures.load()


def _condition(change_rate: float) -> FdmSimpleStorageCondition:
    mesher = _fixtures.storage_mesher()
    return FdmSimpleStorageCondition(_EXERCISE_TIMES, mesher, _fixtures.spot_price(mesher), change_rate)


def test_is_a_step_condition() -> None:
    assert isinstance(_condition(1.5), StepCondition)


def test_mesher_layout_matches_probe(reference: dict[str, Any]) -> None:
    mesher = _fixtures.storage_mesher()
    assert mesher.layout().size() == reference["storage_size"]
    assert list(mesher.layout().dim()) == reference["storage_dim"]


def test_inner_value_stub_matches_cpp(reference: dict[str, Any]) -> None:
    """A zero-strike call makes the inner value the spot price itself."""
    mesher = _fixtures.storage_mesher()
    values = _fixtures.inner_values(mesher, _fixtures.spot_price(mesher), 0.25)
    _fixtures.assert_array_tight(values, reference, "storage_inner")


def test_axes_harvested_from_the_layout(reference: dict[str, Any]) -> None:
    """The constructor's x_/y_ scan reproduces the per-direction node positions."""
    mesher = _fixtures.storage_mesher()
    layout = mesher.layout()
    xs: list[float] = []
    ys: list[float] = []
    for iterator in layout.iter():
        if iterator.coordinates[1] == 0:
            xs.append(mesher.location(iterator, 0))
        if iterator.coordinates[0] == 0:
            ys.append(mesher.location(iterator, 1))
    _fixtures.assert_array_tight(np.array(xs, dtype=np.float64), reference, "storage_x")
    _fixtures.assert_array_tight(np.array(ys, dtype=np.float64), reference, "storage_y")


def test_no_op_away_from_an_exercise_time(reference: dict[str, Any]) -> None:
    a: Array = _fixtures.ref_array(reference, "storage_seed")
    _condition(1.5).apply_to(a, 0.1)
    _fixtures.assert_array_tight(a, reference, "storage_rate15_no_exercise")


@pytest.mark.parametrize(
    ("change_rate", "t", "key"),
    [
        # 1.5 > grid step 1.0: the intermediate-volume scan runs and every
        # interpolation query lands strictly between grid nodes.
        (1.5, 0.25, "storage_rate15_t025"),
        (1.5, 0.5, "storage_rate15_t050"),
        # 0.4 < grid step: the scan loop body never executes.
        (0.4, 0.25, "storage_rate04_t025"),
        # 2.0 == exactly two grid steps: upper_bound / strict-less boundary
        # handling lands on grid nodes.
        (2.0, 0.25, "storage_rate20_t025"),
    ],
)
def test_inject_withdraw_decision(reference: dict[str, Any], change_rate: float, t: float, key: str) -> None:
    a: Array = _fixtures.ref_array(reference, "storage_seed")
    _condition(change_rate).apply_to(a, t)
    _fixtures.assert_array_tight(a, reference, key)


def test_zero_change_rate_is_a_no_op(reference: dict[str, Any]) -> None:
    """With ``changeRate == 0`` nothing can move, so the value must stand.

    This is the sharpest single regression case in the block: it fails the
    moment the bilinear interpolant does not reproduce the grid values exactly
    at the grid nodes.
    """
    a: Array = _fixtures.ref_array(reference, "storage_seed")
    _condition(0.0).apply_to(a, 0.25)
    _fixtures.assert_array_tight(a, reference, "storage_rate00_t025")
    _fixtures.assert_array_tight(a, reference, "storage_seed")


def test_apply_to_mutates_in_place(reference: dict[str, Any]) -> None:
    """C++ ``a = retVal`` writes back into the caller's array."""
    a: Array = _fixtures.ref_array(reference, "storage_seed")
    original = a
    _condition(1.5).apply_to(a, 0.25)
    assert a is original
    _fixtures.assert_array_tight(a, reference, "storage_rate15_t025")


def test_inconsistent_array_dimensions_raises() -> None:
    bad: Array = np.zeros(3, dtype=np.float64)
    with pytest.raises(LibraryException):
        _condition(1.5).apply_to(bad, 0.25)
