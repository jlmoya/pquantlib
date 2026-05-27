"""Nelder-Mead simplex optimization method (scipy-backed).

# C++ parity: ql/math/optimization/simplex.{hpp,cpp} (v1.42.1).

The C++ implementation is a hand-rolled Nelder-Mead following
*Numerical Recipes in C* (2nd ed., chapter 10). The pquantlib port
delegates to ``scipy.optimize.minimize(method='Nelder-Mead', ...)``,
which is also a textbook Nelder-Mead implementation. The two
trajectories are not bit-identical (different starting-simplex
construction details, different ordering of reflect/expand/contract
moves on ties) but the converged minimum agrees to LOOSE tier or
tighter on well-posed problems.

Divergences and approximations:

- **scipy wrapper.** Same caveat as ``LevenbergMarquardt`` —
  cross-validation tests check the converged ``x`` and ``f``, not
  the iteration trajectory.
- **Starting simplex.** C++ constructs an N+1 simplex with
  ``P_0 = x0`` and ``P_i = x0 + lambda * e_i``. scipy's
  ``Nelder-Mead`` accepts an explicit ``initial_simplex`` of shape
  ``(N+1, N)``; pquantlib builds that simplex to match the C++
  construction exactly. This means the two methods see the same
  starting points even though the per-iteration moves diverge.
- **Constraint handling.** C++ Simplex projects the iterate back
  onto the feasible set via ``Constraint::update``; scipy's
  Nelder-Mead has no native constraint support. pquantlib injects
  a large penalty in the cost callback when the iterate violates
  the constraint, matching the C++ infeasible-cost convention.
- **EndCriteria.** scipy returns status 0 on success, 1 on max
  iterations, 2 on max function evaluations. pquantlib translates:

  - status 0 -> ``StationaryPoint`` (the simplex has shrunk
    below ``xatol`` AND ``fatol``, which is the closest semantic
    analogue to C++'s "stationary simplex")
  - status 1 -> ``MaxIterations``
  - status 2 -> ``MaxIterations`` (max-fnev is also an iteration
    cap from the caller's perspective)
  - anything else -> ``Unknown``

  The C++ probe lands at ``StationaryPoint`` (2); pquantlib lands
  at the same Type. Tests cross-validate the convergent ``x`` and
  the success ``Type`` value.
- **EndCriteria parameters.** ``maxIterations`` -> scipy ``maxiter``
  AND ``maxfev`` (both bounded by the same input);
  ``rootEpsilon`` -> ``xatol``; ``functionEpsilon`` -> ``fatol``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pquantlib.math.optimization.end_criteria import Type
from pquantlib.math.optimization.optimization_method import OptimizationMethod

if TYPE_CHECKING:
    from pquantlib.math.optimization.end_criteria import EndCriteria
    from pquantlib.math.optimization.problem import Problem


# Large finite penalty for infeasible iterates (cf. levenberg_marquardt).
_INFEASIBLE_PENALTY: Final[float] = 1e30


class Simplex(OptimizationMethod):
    """Nelder-Mead downhill simplex optimization.

    # C++ parity: ``class Simplex`` in
    # ql/math/optimization/simplex.hpp:58-72 (v1.42.1).

    Minimizes ``f(x) = problem.cost_function.value`` directly (i.e.
    treating the cost function as scalar — Simplex never inspects
    the residual vector). Delegates to scipy's
    ``minimize(method='Nelder-Mead', ...)``.
    """

    __slots__ = ("_lambda",)

    def __init__(self, lambda_: float = 1.0) -> None:
        # C++ parity: simplex.hpp:61 — ctor takes the initial simplex
        # characteristic edge length lambda.
        self._lambda: float = lambda_

    @property
    def lambda_(self) -> float:
        """Characteristic edge length of the initial simplex.

        # C++ parity: simplex.hpp:63 — ``Real lambda() const``.
        """
        return self._lambda

    def minimize(self, problem: Problem, end_criteria: EndCriteria) -> Type:
        # C++ parity: simplex.cpp — minimize() body.
        x0: npt.NDArray[np.float64] = problem.current_value.astype(np.float64, copy=True)
        n = int(x0.size)

        def cost(x: npt.NDArray[np.float64]) -> float:
            # Penalty-based constraint handling. See ``levenberg_marquardt``
            # module docstring for rationale.
            if not problem.constraint.test(x):
                return _INFEASIBLE_PENALTY
            return problem.value(x)

        # Build the initial simplex exactly as C++ does:
        # vertex 0 is x0; vertex i (for i in 1..n) is x0 + lambda*e_{i-1}.
        initial_simplex = np.tile(x0, (n + 1, 1))
        for i in range(n):
            initial_simplex[i + 1, i] += self._lambda

        result = cast(
            "Any",
            minimize(
                cost,
                x0,
                method="Nelder-Mead",
                options={
                    "xatol": end_criteria.root_epsilon,
                    "fatol": end_criteria.function_epsilon,
                    "maxiter": end_criteria.max_iterations,
                    "maxfev": end_criteria.max_iterations,
                    "initial_simplex": initial_simplex,
                    "adaptive": False,
                },
            ),
        )

        problem.set_current_value(np.asarray(result.x, dtype=np.float64))
        problem.set_function_value(float(result.fun))

        return _translate_simplex_status(int(result.status))


def _translate_simplex_status(status: int) -> Type:
    """Translate ``scipy.optimize.minimize`` Nelder-Mead status -> ``Type``.

    Mapping (see scipy docs):
    - 0: converged below xatol+fatol -> StationaryPoint
    - 1: max iterations reached -> MaxIterations
    - 2: max function evaluations reached -> MaxIterations
    - anything else -> Unknown
    """
    if status == 0:
        return Type.StationaryPoint
    if status in {1, 2}:
        return Type.MaxIterations
    return Type.Unknown
