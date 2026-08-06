"""Cross-validation tests for ``FdmSnapshotCondition``.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmsnapshotcondition.{hpp,cpp}
# @ v1.43.

Reference: ``migration-harness/references/v143/methods/stepconditions.json``,
``snap_*`` keys.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.step_conditions.fdm_snapshot_condition import (
    FdmSnapshotCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)
from pquantlib.testing.tolerance import exact
from tests.methods.finitedifferences.step_conditions import _fixtures


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return _fixtures.load()


def test_is_a_step_condition() -> None:
    assert isinstance(FdmSnapshotCondition(1.5), StepCondition)


def test_get_time(reference: dict[str, Any]) -> None:
    """EXACT: the snapshot time is stored, not computed."""
    cond = FdmSnapshotCondition(1.5)
    exact(cond.get_time(), float(reference["snap_time"]))


def test_values_empty_before_any_snapshot(reference: dict[str, Any]) -> None:
    cond = FdmSnapshotCondition(1.5)
    assert cond.get_values().size == reference["snap_values_size_initial"]


def test_no_snapshot_at_a_different_time(reference: dict[str, Any]) -> None:
    cond = FdmSnapshotCondition(1.5)
    a: Array = _fixtures.ref_array(reference, "snap_seed")
    cond.apply_to(a, 0.5)
    assert cond.get_values().size == reference["snap_values_size_after_wrong_t"]
    # The condition never modifies the solution vector.
    _fixtures.assert_array_tight(a, reference, "snap_a_after_wrong_t")


def test_snapshot_at_exact_time(reference: dict[str, Any]) -> None:
    """EXACT: a snapshot is a copy — no arithmetic can perturb it."""
    cond = FdmSnapshotCondition(1.5)
    a: Array = _fixtures.ref_array(reference, "snap_seed")
    cond.apply_to(a, 1.5)
    assert cond.get_values().size == reference["snap_values_size_at_t"]
    expected: list[float] = reference["snap_values_at_t"]
    for i, exp in enumerate(expected):
        exact(float(cond.get_values()[i]), float(exp), reason=f"snap_values_at_t[{i}]")
    # The array itself is untouched.
    _fixtures.assert_array_tight(a, reference, "snap_a_at_t")


def test_snapshot_is_a_copy_not_an_alias(reference: dict[str, Any]) -> None:
    """C++ ``values_ = a`` copies; mutating ``a`` afterwards must not leak in."""
    cond = FdmSnapshotCondition(1.5)
    a: Array = _fixtures.ref_array(reference, "snap_seed")
    cond.apply_to(a, 1.5)
    a[0] = 99.0
    a[3] = -7.0
    cond.apply_to(a, 2.5)  # not the snapshot time -> stored values must survive
    expected: list[float] = reference["snap_values_after_mutation"]
    for i, exp in enumerate(expected):
        exact(float(cond.get_values()[i]), float(exp), reason=f"snap_values_after_mutation[{i}]")


def test_snapshot_rearms_on_a_second_hit(reference: dict[str, Any]) -> None:
    """Applying again exactly at ``t`` overwrites the stored values."""
    cond = FdmSnapshotCondition(1.5)
    a: Array = _fixtures.ref_array(reference, "snap_seed")
    cond.apply_to(a, 1.5)
    a[0] = 99.0
    a[3] = -7.0
    cond.apply_to(a, 2.5)
    cond.apply_to(a, 1.5)
    expected: list[float] = reference["snap_values_rearmed"]
    for i, exp in enumerate(expected):
        exact(float(cond.get_values()[i]), float(exp), reason=f"snap_values_rearmed[{i}]")


def test_time_test_is_exact_not_tolerant(reference: dict[str, Any]) -> None:
    """C++ tests ``t == t_`` with no tolerance; one ULP off must not snapshot."""
    cond = FdmSnapshotCondition(1.5)
    a: Array = _fixtures.ref_array(reference, "snap_seed")
    cond.apply_to(a, math.nextafter(1.5, 2.0))
    assert cond.get_values().size == 0
