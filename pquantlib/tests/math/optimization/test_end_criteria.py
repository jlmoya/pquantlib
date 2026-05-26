"""EndCriteria dataclass + Type IntEnum behavioral tests."""

from __future__ import annotations

import dataclasses

import pytest

from pquantlib.math.optimization.end_criteria import EndCriteria, Type


def test_type_enum_integer_values_mirror_cpp() -> None:
    # C++ parity: endcriteria.hpp:42-49 — integer ordering of Type.
    assert Type.None_ == 0
    assert Type.MaxIterations == 1
    assert Type.StationaryPoint == 2
    assert Type.StationaryFunctionValue == 3
    assert Type.StationaryFunctionAccuracy == 4
    assert Type.ZeroGradientNorm == 5
    assert Type.FunctionEpsilon == 6
    assert Type.Unknown == 7


def test_type_enum_count_is_eight() -> None:
    assert len(Type) == 8


def test_end_criteria_construction() -> None:
    ec = EndCriteria(
        max_iterations=1000,
        max_stationary_state=100,
        root_epsilon=1e-8,
        function_epsilon=1e-9,
        gradient_norm_epsilon=1e-7,
    )
    assert ec.max_iterations == 1000
    assert ec.max_stationary_state == 100
    assert ec.root_epsilon == 1e-8
    assert ec.function_epsilon == 1e-9
    assert ec.gradient_norm_epsilon == 1e-7


def test_end_criteria_is_frozen() -> None:
    ec = EndCriteria(
        max_iterations=1000,
        max_stationary_state=100,
        root_epsilon=1e-8,
        function_epsilon=1e-9,
        gradient_norm_epsilon=1e-7,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ec.max_iterations = 2000  # type: ignore[misc]


def test_end_criteria_equality_by_value() -> None:
    a = EndCriteria(1000, 100, 1e-8, 1e-9, 1e-7)
    b = EndCriteria(1000, 100, 1e-8, 1e-9, 1e-7)
    c = EndCriteria(1001, 100, 1e-8, 1e-9, 1e-7)
    assert a == b
    assert a != c
