"""EndCriteria dataclass + Type IntEnum behavioral tests."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.optimization.end_criteria import NULL_REAL, NULL_SIZE, EndCriteria, Type
from pquantlib.testing import reference_reader


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


# --- v1.43 additions: the stateful checkers + the constructor ---------------


def _cpp_optimization() -> dict[str, Any]:
    return reference_reader.load("v143/math/optimization")


def test_null_sentinels_match_cpp() -> None:
    """``Null<Real>()`` is ``FLT_MAX`` and ``Null<Size>()`` is ``INT_MAX``.

    # C++ parity: ql/utilities/null.hpp — NOT the max of the type itself.
    """
    cpp = _cpp_optimization()
    assert cpp["ec_null_real_sentinel"] == NULL_REAL
    assert cpp["ec_null_size_sentinel"] == NULL_SIZE


def test_null_max_stationary_state_becomes_min_half_maxiter_and_100() -> None:
    """# C++ parity: endcriteria.cpp:40-42."""
    cpp = _cpp_optimization()
    assert (
        EndCriteria(1000, NULL_SIZE, 1e-8, 1e-9, 1e-7).max_stationary_state
        == cpp["ec_null_stat_1000"]
    )
    assert (
        EndCriteria(30, NULL_SIZE, 1e-8, 1e-9, 1e-7).max_stationary_state
        == cpp["ec_null_stat_30"]
    )


def test_null_gradient_norm_epsilon_falls_back_to_function_epsilon() -> None:
    """# C++ parity: endcriteria.cpp:52-53."""
    cpp = _cpp_optimization()
    ec = EndCriteria(1000, 100, 1e-8, 1.25e-9, NULL_REAL)
    assert ec.gradient_norm_epsilon == cpp["ec_null_gradeps"]
    assert ec.gradient_norm_epsilon == ec.function_epsilon


@pytest.mark.parametrize(
    ("max_iterations", "max_stationary_state", "message"),
    [
        (100, 1, "greater than one"),
        (100, 0, "greater than one"),
        (10, 10, "must be less than"),
        (10, 20, "must be less than"),
    ],
)
def test_constructor_requirements(
    max_iterations: int, max_stationary_state: int, message: str
) -> None:
    """# C++ parity: endcriteria.cpp:43-51 — two ``QL_REQUIRE``s."""
    with pytest.raises(LibraryException, match=message):
        EndCriteria(max_iterations, max_stationary_state, 1e-8, 1e-9, 1e-7)


def test_check_stationary_point_counter_sequence_matches_cpp() -> None:
    """The in-out ``statStateIterations`` counter, step by step.

    # C++ parity: endcriteria.cpp:64-77. A move of at least ``rootEpsilon``
    # RESETS the counter; the criterion fires only once it strictly exceeds
    # ``maxStationaryStateIterations``.
    """
    cpp = _cpp_optimization()
    ec = EndCriteria(100, 3, 0.1, 0.01, 1e-7)
    stat = 0
    fired: list[int] = []
    counters: list[int] = []
    types: list[int] = []
    current = Type.None_
    for x in cpp["ec_csp_x"]:  # type: ignore[union-attr]
        stat, hit = ec.check_stationary_point(0.0, float(x), stat)
        if hit is not None:
            current = hit
        fired.append(1 if hit is not None else 0)
        counters.append(stat)
        types.append(int(current))
    assert fired == cpp["ec_csp_fired"]
    assert counters == cpp["ec_csp_counter"]
    assert types == cpp["ec_csp_type"]


def test_check_stationary_function_value_counter_sequence_matches_cpp() -> None:
    """# C++ parity: endcriteria.cpp:79-93."""
    cpp = _cpp_optimization()
    ec = EndCriteria(100, 2, 0.1, 0.05, 1e-7)
    stat = 0
    fired: list[int] = []
    counters: list[int] = []
    types: list[int] = []
    current = Type.None_
    for f in cpp["ec_csfv_f"]:  # type: ignore[union-attr]
        stat, hit = ec.check_stationary_function_value(0.0, float(f), stat)
        if hit is not None:
            current = hit
        fired.append(1 if hit is not None else 0)
        counters.append(stat)
        types.append(int(current))
    assert fired == cpp["ec_csfv_fired"]
    assert counters == cpp["ec_csfv_counter"]
    assert types == cpp["ec_csfv_type"]


def test_seeding_the_counter_with_the_maximum_fires_unconditionally() -> None:
    """The idiom ``Simplex`` and ``LineSearchBasedMethod`` rely on.

    Both pass ``endCriteria.maxStationaryStateIterations()`` AS the counter,
    which makes the pre-increment exceed the maximum on the first call.
    """
    cpp = _cpp_optimization()
    ec = EndCriteria(100, 7, 0.1, 0.05, 1e-7)
    _, hit = ec.check_stationary_point(0.0, 0.0, ec.max_stationary_state)
    assert bool(cpp["ec_seeded_csp_fired"])
    assert hit is not None
    assert int(hit) == cpp["ec_seeded_csp_type"]
    _, hit2 = ec.check_stationary_function_value(0.0, 0.0, ec.max_stationary_state)
    assert bool(cpp["ec_seeded_csfv_fired"])
    assert hit2 is not None
    assert int(hit2) == cpp["ec_seeded_csfv_type"]


def test_check_stationary_function_accuracy_matches_cpp() -> None:
    """# C++ parity: endcriteria.cpp:95-105 — gated on ``positiveOptimization``."""
    cpp = _cpp_optimization()
    ec = EndCriteria(100, 3, 0.1, 0.05, 1e-7)
    assert (ec.check_stationary_function_accuracy(0.01, False) is not None) is bool(
        cpp["ec_csfa_negative_optim"]
    )
    hit = ec.check_stationary_function_accuracy(0.01, True)
    assert bool(cpp["ec_csfa_below"])
    assert hit is not None
    assert int(hit) == cpp["ec_csfa_below_type"]
    assert (ec.check_stationary_function_accuracy(0.05, True) is not None) is bool(
        cpp["ec_csfa_at_epsilon"]
    )


def test_operator_call_short_circuit_chain_matches_cpp() -> None:
    """# C++ parity: endcriteria.cpp:125-138 — max-iter, stationary f, accuracy, |g|."""
    cpp = _cpp_optimization()
    ec = EndCriteria(10, 2, 0.1, 0.05, 1e-3)
    stat = 0
    fired: list[int] = []
    counters: list[int] = []
    types: list[int] = []
    fnews = [0.5, 0.99, 0.99, 0.99, 0.99]
    gnews = [1.0, 1.0, 1.0, 1.0, 1e-6]
    for i in range(5):
        current = Type.None_
        stat, hit = ec(i, stat, False, 1.0, 0.0, fnews[i], gnews[i])
        if hit is not None:
            current = hit
        fired.append(1 if hit is not None else 0)
        counters.append(stat)
        types.append(int(current))
    assert fired == cpp["ec_call_fired"]
    assert counters == cpp["ec_call_counter"]
    assert types == cpp["ec_call_type"]

    _, hit_max = ec(10, 0, False, 1.0, 0.0, 0.5, 1.0)
    assert bool(cpp["ec_call_maxiter_fired"])
    assert hit_max is not None
    assert int(hit_max) == cpp["ec_call_maxiter_type"]

    _, hit_acc = ec(0, 0, True, 1.0, 0.0, 0.01, 1.0)
    assert bool(cpp["ec_call_accuracy_fired"])
    assert hit_acc is not None
    assert int(hit_acc) == cpp["ec_call_accuracy_type"]


def test_succeeded_matches_cpp() -> None:
    """# C++ parity: endcriteria.cpp:161-165 — ZeroGradientNorm is NOT a success."""
    cpp = _cpp_optimization()
    flags = [1 if EndCriteria.succeeded(Type(i)) else 0 for i in range(8)]
    assert flags == cpp["ec_succeeded_by_type"]
    assert EndCriteria.succeeded(Type.ZeroGradientNorm) is False
