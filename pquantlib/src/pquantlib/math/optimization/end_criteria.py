"""Optimization end criteria + outcome enum.

# C++ parity: ql/math/optimization/endcriteria.hpp +
# endcriteria.cpp (v1.42.1).

``EndCriteria`` bundles the five thresholds that every QuantLib
optimization method consults: max iterations, max stationary-state
iterations (i.e. how long a residual may stay flat before declaring
convergence), root epsilon (x-variation), function epsilon
(y-variation), and gradient-norm epsilon. ``Type`` is the discrete
outcome of an optimization run.

L1-D ported the dataclass + enum only. ``checkMaxIterations`` and
``checkZeroGradientNorm`` follow here: they are the two checks
``LBFGSB`` consults, and the LBFGSB probe cross-validates both (one
case stops on each). The remaining checkers (``operator()``,
``checkStationaryPoint``, ``checkStationaryFunctionValue``,
``checkStationaryFunctionAccuracy``, ``succeeded``) stay deferred —
they carry the ``statStateIterations`` in-out counter and no ported
method calls them yet.

C++ signals a fired criterion through an ``EndCriteria::Type&``
out-parameter plus a ``bool`` return. Python has no out-parameters, so
the checkers return ``Type | None``: the ``Type`` to adopt when the
criterion fires, ``None`` when it does not. Callers read as::

    hit = end_criteria.check_max_iterations(iteration)
    if hit is not None:
        ec_type = hit
        break
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

    def check_max_iterations(self, iteration: int) -> Type | None:
        """``Type.MaxIterations`` once ``iteration`` reaches the cap, else ``None``.

        # C++ parity: endcriteria.cpp:57-63 — ``checkMaxIterations``. The
        # test is ``iteration < maxIterations_`` -> not fired, so the
        # criterion trips on the iteration *index* equalling the cap, i.e.
        # after exactly ``max_iterations`` completed iterations.
        """
        if iteration < self.max_iterations:
            return None
        return Type.MaxIterations

    def check_zero_gradient_norm(self, gradient_norm: float) -> Type | None:
        """``Type.ZeroGradientNorm`` when ``gradient_norm`` is below the epsilon.

        # C++ parity: endcriteria.cpp:110-116 — ``checkZeroGradientNorm``.

        The argument is the gradient norm itself, not its square: callers
        that cache a squared norm (as ``Problem.gradient_norm_value``
        does) must pass the unsquared value here.
        """
        if gradient_norm >= self.gradient_norm_epsilon:
            return None
        return Type.ZeroGradientNorm
