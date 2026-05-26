"""Optimization end criteria + outcome enum.

# C++ parity: ql/math/optimization/endcriteria.hpp +
# endcriteria.cpp (v1.42.1).

``EndCriteria`` bundles the five thresholds that every QuantLib
optimization method consults: max iterations, max stationary-state
iterations (i.e. how long a residual may stay flat before declaring
convergence), root epsilon (x-variation), function epsilon
(y-variation), and gradient-norm epsilon. ``Type`` is the discrete
outcome of an optimization run.

L1-D only ports the dataclass + enum; the C++ checker methods
(``operator()``, ``checkMaxIterations``, ``checkStationaryPoint``,
``checkStationaryFunctionValue``, ``checkStationaryFunctionAccuracy``,
``checkZeroGradientNorm``, ``succeeded``) are deferred — they are
only used by Levenberg-Marquardt / Simplex / BFGS, all carved out
of L1-D.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Type(IntEnum):
    """Optimization termination outcome.

    # C++ parity: ``EndCriteria::Type`` in
    # ql/math/optimization/endcriteria.hpp:42-49 (v1.42.1):
    # ``{None, MaxIterations, StationaryPoint, StationaryFunctionValue,
    # StationaryFunctionAccuracy, ZeroGradientNorm,
    # FunctionEpsilonTooSmall, Unknown}``.

    pquantlib renames ``None`` to ``None_`` (Python keyword collision)
    and ``FunctionEpsilonTooSmall`` to ``FunctionEpsilon`` per the
    L1-D design spec — semantically identical, the integer values
    match C++ exactly so ints can be compared across the boundary.
    """

    None_ = 0
    MaxIterations = 1
    StationaryPoint = 2
    StationaryFunctionValue = 3
    StationaryFunctionAccuracy = 4
    ZeroGradientNorm = 5
    FunctionEpsilon = 6
    Unknown = 7


@dataclass(frozen=True, slots=True)
class EndCriteria:
    """Stop-condition bundle for optimization methods.

    # C++ parity: ``class EndCriteria`` in
    # ql/math/optimization/endcriteria.hpp:40-108 (v1.42.1).

    The C++ version is a class with mutable
    ``maxStationaryStateIterations_``; the Python port is frozen
    because the checker methods (``operator()`` & co.) that mutate
    that field are deferred to a later cluster — until then the
    semantics are pure value-bundle.
    """

    max_iterations: int
    max_stationary_state: int
    root_epsilon: float
    function_epsilon: float
    gradient_norm_epsilon: float
