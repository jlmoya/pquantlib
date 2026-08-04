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


# --- checkers ---------------------------------------------------------------


def test_check_max_iterations_fires_only_at_the_cap() -> None:
    # C++ parity: endcriteria.cpp:57-63 — ``iteration < maxIterations_``
    # means "not fired", so the trip happens AT the cap, not past it.
    ec = EndCriteria(5, 2, 1e-8, 1e-9, 1e-7)
    assert ec.check_max_iterations(0) is None
    assert ec.check_max_iterations(4) is None
    assert ec.check_max_iterations(5) is Type.MaxIterations
    assert ec.check_max_iterations(6) is Type.MaxIterations


def test_check_zero_gradient_norm_fires_strictly_below_the_epsilon() -> None:
    # C++ parity: endcriteria.cpp:110-116 — ``gradientNorm >= eps`` means
    # "not fired", so equality does NOT trip the criterion.
    ec = EndCriteria(1000, 100, 1e-8, 1e-9, 1e-7)
    assert ec.check_zero_gradient_norm(1e-6) is None
    assert ec.check_zero_gradient_norm(1e-7) is None
    assert ec.check_zero_gradient_norm(9.9e-8) is Type.ZeroGradientNorm
    assert ec.check_zero_gradient_norm(0.0) is Type.ZeroGradientNorm
