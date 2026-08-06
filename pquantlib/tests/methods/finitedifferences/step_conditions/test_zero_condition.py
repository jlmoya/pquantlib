"""Cross-validation tests for ``ZeroCondition``.

# C++ parity: ql/methods/finitedifferences/zerocondition.hpp @ v1.43.

The class lives one directory up (``methods/finitedifferences/zero_condition``)
because that is where its C++ header sits, but it belongs to the
step-conditions cluster, so its tests live here with the rest.

Reference: ``migration-harness/references/v143/methods/stepconditions.json``,
``zero_*`` keys.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)
from pquantlib.methods.finitedifferences.zero_condition import ZeroCondition
from pquantlib.testing.tolerance import exact
from tests.methods.finitedifferences.step_conditions import _fixtures


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return _fixtures.load()


def test_zero_condition_is_a_step_condition() -> None:
    assert isinstance(ZeroCondition(), StepCondition)


def test_apply_to_floors_at_zero(reference: dict[str, Any]) -> None:
    """EXACT: ``max(a[i], 0.0)`` is selection, not arithmetic — no rounding."""
    a: Array = _fixtures.ref_array(reference, "zero_seed")
    ZeroCondition().apply_to(a, 0.42)
    expected: list[float] = reference["zero_out_t042"]
    for i, exp in enumerate(expected):
        exact(float(a[i]), float(exp), reason=f"zero_out_t042[{i}]")


def test_apply_to_ignores_time(reference: dict[str, Any]) -> None:
    """C++ leaves the ``Time`` parameter unnamed: the result cannot depend on it."""
    a: Array = _fixtures.ref_array(reference, "zero_seed")
    ZeroCondition().apply_to(a, 0.0)
    expected: list[float] = reference["zero_out_t000"]
    for i, exp in enumerate(expected):
        exact(float(a[i]), float(exp), reason=f"zero_out_t000[{i}]")


def test_apply_to_is_idempotent(reference: dict[str, Any]) -> None:
    a: Array = _fixtures.ref_array(reference, "zero_seed")
    cond = ZeroCondition()
    cond.apply_to(a, 0.42)
    cond.apply_to(a, 0.42)
    expected: list[float] = reference["zero_out_twice"]
    for i, exp in enumerate(expected):
        exact(float(a[i]), float(exp), reason=f"zero_out_twice[{i}]")


def test_apply_to_mutates_in_place(reference: dict[str, Any]) -> None:
    """The C++ signature takes ``Array&``; the caller's buffer must be the one written."""
    a: Array = _fixtures.ref_array(reference, "zero_seed")
    original = a
    ZeroCondition().apply_to(a, 1.0)
    assert a is original
    assert float(a[0]) == 0.0


def test_negative_zero_survives() -> None:
    """``std::max(-0.0, 0.0)`` returns ``-0.0`` because ``-0.0 < 0.0`` is false.

    Signed zero cannot round-trip through the probe's JSON (``-0`` parses back
    as ``0`` in Python), so this is pinned against the C++ *specification* of
    ``std::max`` — ``a < b ? b : a`` — rather than against an emitted value.
    ``numpy.maximum`` does not agree here, which is exactly why the port spells
    the comparison out.
    """
    a: Array = np.array([-0.0], dtype=np.float64)
    ZeroCondition().apply_to(a, 1.0)
    assert math.copysign(1.0, float(a[0])) == -1.0


def test_nan_survives() -> None:
    """``std::max(NaN, 0.0)`` returns NaN: ``NaN < 0.0`` is false.

    Same reasoning as the signed-zero case — pinned against the C++ contract,
    not against an emitted value (JSON has no NaN literal).
    """
    a: Array = np.array([math.nan], dtype=np.float64)
    ZeroCondition().apply_to(a, 1.0)
    assert math.isnan(float(a[0]))
