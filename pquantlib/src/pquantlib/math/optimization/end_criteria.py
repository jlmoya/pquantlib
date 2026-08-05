"""Optimization end criteria + outcome enum.

# C++ parity: ql/math/optimization/endcriteria.{hpp,cpp} (v1.43).

``EndCriteria`` bundles the five thresholds that every QuantLib
optimization method consults: max iterations, max stationary-state
iterations (i.e. how long a residual may stay flat before declaring
convergence), root epsilon (x-variation), function epsilon
(y-variation), and gradient-norm epsilon. ``Type`` is the discrete
outcome of an optimization run.

C++ signals a fired criterion through an ``EndCriteria::Type&``
out-parameter plus a ``bool`` return. Python has no out-parameters, so:

- the STATELESS checkers (``check_max_iterations``,
  ``check_zero_gradient_norm``, ``check_stationary_function_accuracy``)
  return ``Type | None``: the ``Type`` to adopt when the criterion
  fires, ``None`` when it does not;
- the STATEFUL checkers (``check_stationary_point``,
  ``check_stationary_function_value``, ``__call__``) additionally carry
  the C++ ``Size& statStateIterations`` in-out counter, so they return
  ``(stat_state_iterations, Type | None)``.

Callers read as::

    stat, hit = end_criteria.check_stationary_point(x_old, x_new, stat)
    if hit is not None:
        ec_type = hit

Returning ``None`` (rather than the previous type) preserves the C++
semantics exactly: ``ecType`` is an in-out parameter that keeps its
previous value when nothing fires.

The "seed the counter with the maximum" idiom
---------------------------------------------
``Simplex::minimize`` and ``LineSearchBasedMethod::minimize`` both pass
``endCriteria.maxStationaryStateIterations()`` AS the counter argument
(simplex.cpp:151, linesearchbasedmethod.cpp:100). Since the checker
pre-increments the counter and then tests ``counter <= max``, seeding it
with ``max`` makes the criterion fire unconditionally. That is not a
bug to smooth over — it is how those two methods manufacture their
``StationaryPoint`` / ``StationaryFunctionValue`` return value, and the
ports reproduce it verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

from pquantlib import qassert

# C++ parity: ql/utilities/null.hpp — ``Null<T>()`` yields
# ``numeric_limits<float>::max()`` for floating-point T and
# ``numeric_limits<int>::max()`` for integral T (NOT the max of T itself).
NULL_REAL: Final[float] = 3.4028234663852886e38
NULL_SIZE: Final[int] = 2147483647


class Type(IntEnum):
    """Optimization termination outcome.

    # C++ parity: ``EndCriteria::Type`` in
    # ql/math/optimization/endcriteria.hpp:42-49 (v1.43):
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
    # ql/math/optimization/endcriteria.{hpp,cpp} (v1.43).

    The C++ class declares ``maxStationaryStateIterations_`` ``mutable``
    but only the constructor ever writes it, so the Python port stays a
    frozen value bundle; ``__post_init__`` applies the constructor's
    ``Null`` substitutions through ``object.__setattr__``.
    """

    max_iterations: int
    max_stationary_state: int
    root_epsilon: float
    function_epsilon: float
    gradient_norm_epsilon: float

    def __post_init__(self) -> None:
        """Apply the C++ constructor's ``Null`` defaults and requirements.

        # C++ parity: endcriteria.cpp:29-54.
        """
        max_stat = self.max_stationary_state
        if max_stat == NULL_SIZE:
            max_stat = min(self.max_iterations // 2, 100)
            object.__setattr__(self, "max_stationary_state", max_stat)
        qassert.require(
            max_stat > 1,
            f"maxStationaryStateIterations_ ({max_stat}) must be greater than one",
        )
        qassert.require(
            max_stat < self.max_iterations,
            f"maxStationaryStateIterations_ ({max_stat}) must be less than "
            f"maxIterations_ ({self.max_iterations})",
        )
        if self.gradient_norm_epsilon == NULL_REAL:
            object.__setattr__(self, "gradient_norm_epsilon", self.function_epsilon)

    # --- stateless checkers ---------------------------------------------

    def check_max_iterations(self, iteration: int) -> Type | None:
        """``Type.MaxIterations`` once ``iteration`` reaches the cap, else ``None``.

        # C++ parity: endcriteria.cpp:56-62 — ``checkMaxIterations``. The
        # test is ``iteration < maxIterations_`` -> not fired, so the
        # criterion trips on the iteration *index* equalling the cap, i.e.
        # after exactly ``max_iterations`` completed iterations.
        """
        if iteration < self.max_iterations:
            return None
        return Type.MaxIterations

    def check_zero_gradient_norm(self, gradient_norm: float) -> Type | None:
        """``Type.ZeroGradientNorm`` when ``gradient_norm`` is below the epsilon.

        # C++ parity: endcriteria.cpp:117-123 — ``checkZeroGradientNorm``.

        The argument is whatever the caller chose to call "the gradient
        norm": ``EndCriteria::operator()`` is handed ``normgnew``, and
        ``LBFGSB`` feeds it the projected-gradient infinity norm. No
        squaring is applied here either way.
        """
        if gradient_norm >= self.gradient_norm_epsilon:
            return None
        return Type.ZeroGradientNorm

    def check_stationary_function_accuracy(
        self, f: float, positive_optimization: bool
    ) -> Type | None:
        """``Type.StationaryFunctionAccuracy`` when ``f`` is below the epsilon.

        # C++ parity: endcriteria.cpp:95-105 —
        # ``checkStationaryFunctionAccuracy``. Only meaningful for an
        # objective known to be non-negative, hence the
        # ``positive_optimization`` gate: when it is false the criterion
        # never fires.
        """
        if not positive_optimization:
            return None
        if f >= self.function_epsilon:
            return None
        return Type.StationaryFunctionAccuracy

    # --- stateful checkers (carry the statStateIterations counter) -------

    def check_stationary_point(
        self, x_old: float, x_new: float, stat_state_iterations: int
    ) -> tuple[int, Type | None]:
        """Root-variation check; returns the updated counter and the outcome.

        # C++ parity: endcriteria.cpp:64-77 — ``checkStationaryPoint``.

        A move of at least ``root_epsilon`` RESETS the counter to zero and
        reports nothing. A smaller move increments it, and the criterion
        only fires once the counter strictly exceeds
        ``max_stationary_state``.
        """
        if abs(x_new - x_old) >= self.root_epsilon:
            return 0, None
        stat_state_iterations += 1
        if stat_state_iterations <= self.max_stationary_state:
            return stat_state_iterations, None
        return stat_state_iterations, Type.StationaryPoint

    def check_stationary_function_value(
        self, fx_old: float, fx_new: float, stat_state_iterations: int
    ) -> tuple[int, Type | None]:
        """Function-variation check; returns the updated counter and the outcome.

        # C++ parity: endcriteria.cpp:79-93 —
        # ``checkStationaryFunctionValue``. Same shape as
        # ``check_stationary_point`` but against ``function_epsilon``.
        """
        if abs(fx_new - fx_old) >= self.function_epsilon:
            return 0, None
        stat_state_iterations += 1
        if stat_state_iterations <= self.max_stationary_state:
            return stat_state_iterations, None
        return stat_state_iterations, Type.StationaryFunctionValue

    def __call__(
        self,
        iteration: int,
        stat_state_iterations: int,
        positive_optimization: bool,
        fold: float,
        normgold: float,
        fnew: float,
        normgnew: float,
    ) -> tuple[int, Type | None]:
        """Composite check — the C++ ``operator()``.

        # C++ parity: endcriteria.cpp:125-138. Short-circuiting ``||``
        # chain: max-iterations, then stationary function value, then
        # stationary function accuracy, then zero gradient norm. The
        # ``normgold`` argument is accepted and ignored, exactly as in C++
        # (the parameter is left unnamed there).
        """
        del normgold  # C++ leaves this parameter unnamed: accepted, unused.
        hit = self.check_max_iterations(iteration)
        if hit is not None:
            return stat_state_iterations, hit
        stat_state_iterations, hit = self.check_stationary_function_value(
            fold, fnew, stat_state_iterations
        )
        if hit is not None:
            return stat_state_iterations, hit
        hit = self.check_stationary_function_accuracy(fnew, positive_optimization)
        if hit is not None:
            return stat_state_iterations, hit
        return stat_state_iterations, self.check_zero_gradient_norm(normgnew)

    @staticmethod
    def succeeded(ec_type: Type) -> bool:
        """True for the three outcomes C++ considers a successful stop.

        # C++ parity: endcriteria.cpp:161-165 — ``succeeded``. Note that
        # ``ZeroGradientNorm`` is deliberately NOT in the list.
        """
        return ec_type in {
            Type.StationaryPoint,
            Type.StationaryFunctionValue,
            Type.StationaryFunctionAccuracy,
        }
