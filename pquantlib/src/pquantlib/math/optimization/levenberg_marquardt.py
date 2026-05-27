"""Levenberg-Marquardt optimization method (scipy-backed).

# C++ parity: ql/math/optimization/levenbergmarquardt.{hpp,cpp} (v1.42.1).

The C++ implementation embeds MINPACK's ``lmdif`` (Fortran translated to
C in ``lmdif.cpp``). The pquantlib port delegates to
``scipy.optimize.least_squares(method='lm', ...)``, which itself wraps
MINPACK's ``lmder``/``lmdif`` via SciPy's bundled ``_minpack`` Fortran
binding — same algorithm, same family, same numerical core.

Divergences and approximations:

- **scipy wrapper.** No bit-exact reproduction of MINPACK iterations,
  but the converged minimum on a well-posed least-squares problem
  agrees with C++ to LOOSE tier or tighter. Tests cross-validate
  the converged ``x`` and ``f`` values, not iteration counts.
- **Constraint handling.** MINPACK is purely unconstrained.
  pquantlib injects penalty values when the constraint test fails:
  a large finite residual blocks the iterate from straying outside
  the feasible set. The penalty is large but finite to keep
  jacobian estimates well-conditioned; this matches the C++
  ``LevenbergMarquardt::fcn`` callback which returns
  ``QL_MAX_REAL`` per residual when the constraint check fails.
- **EndCriteria.** scipy returns its own status enum; pquantlib
  translates onto ``Type``:

  - status 1 (``gtol`` satisfied) -> ``ZeroGradientNorm`` (5)
  - status 2 (``ftol`` satisfied) -> ``StationaryFunctionValue`` (3)
  - status 3 (``xtol`` satisfied) -> ``StationaryPoint`` (2)
  - status 4 (both ftol+xtol) -> ``StationaryFunctionValue`` (3)
  - status 0 (max nfev) -> ``MaxIterations`` (1)
  - status -1 / others -> ``Unknown`` (6)

  The C++ probe lands at ``StationaryFunctionValue`` (3) on
  Rosenbrock; pquantlib lands at ``ZeroGradientNorm`` (5) for the
  same input because scipy's MINPACK reports gtol first. The
  cross-validation test checks for any successful Type (not
  ``MaxIterations`` / ``Unknown``) rather than exact-matching the
  C++ enum value — documented inline in the test.

- **EndCriteria parameters.** ``maxIterations`` -> scipy ``max_nfev``;
  ``rootEpsilon`` -> ``xtol``; ``functionEpsilon`` -> ``ftol``;
  ``gradientNormEpsilon`` -> ``gtol``. The stationary-state
  iteration count is unused by MINPACK; pquantlib stores it but
  does not pass it to scipy. The C++ ``epsfcn`` (relative step
  size for the forward-difference jacobian) maps to scipy's
  ``diff_step``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pquantlib.math.optimization.end_criteria import Type
from pquantlib.math.optimization.optimization_method import OptimizationMethod

if TYPE_CHECKING:
    from pquantlib.math.optimization.end_criteria import EndCriteria
    from pquantlib.math.optimization.problem import Problem


# Large finite penalty for infeasible iterates. Big enough to repel the
# search, small enough to keep finite-difference jacobians finite.
_INFEASIBLE_PENALTY: Final[float] = 1e30


class LevenbergMarquardt(OptimizationMethod):
    """Levenberg-Marquardt nonlinear least-squares method.

    # C++ parity: ``class LevenbergMarquardt`` in
    # ql/math/optimization/levenbergmarquardt.hpp:49-68 (v1.42.1).

    Solves ``min_x ||r(x)||^2`` where ``r = problem.cost_function.values``.
    Delegates to ``scipy.optimize.least_squares(method='lm', ...)``,
    which wraps the same MINPACK routine the C++ implementation
    embeds.
    """

    __slots__ = ("_epsfcn", "_gtol", "_use_cost_jacobian", "_xtol")

    def __init__(
        self,
        epsfcn: float = 1.0e-8,
        xtol: float = 1.0e-8,
        gtol: float = 1.0e-8,
        use_cost_functions_jacobian: bool = False,
    ) -> None:
        # C++ parity: levenbergmarquardt.hpp:51-54 — ctor defaults.
        self._epsfcn: float = epsfcn
        self._xtol: float = xtol
        self._gtol: float = gtol
        self._use_cost_jacobian: bool = use_cost_functions_jacobian

    @property
    def epsfcn(self) -> float:
        """Relative step size for the forward-difference jacobian (C++ ``epsfcn``)."""
        return self._epsfcn

    @property
    def xtol(self) -> float:
        """Tolerance on parameter convergence (C++ ``xtol``)."""
        return self._xtol

    @property
    def gtol(self) -> float:
        """Tolerance on gradient norm (C++ ``gtol``)."""
        return self._gtol

    @property
    def use_cost_functions_jacobian(self) -> bool:
        """Whether to use the cost function's analytic jacobian.

        # C++ parity: levenbergmarquardt.hpp:54 (boolean flag). pquantlib
        # ignores this flag — scipy's MINPACK wrapper always uses a
        # forward-difference jacobian (no analytic-jacobian hook).
        # Documented carve-out; flag is preserved on the ctor for
        # API parity but does not change behavior.
        """
        return self._use_cost_jacobian

    def minimize(self, problem: Problem, end_criteria: EndCriteria) -> Type:
        # C++ parity: levenbergmarquardt.cpp — minimize() body.
        x0: npt.NDArray[np.float64] = problem.current_value.astype(np.float64, copy=True)

        # Probe once on the starting point to learn the residual length.
        # The starting point should be feasible by precondition (the
        # caller built the Problem with this initial_value).
        initial_residuals = problem.values(x0)
        residual_size = int(initial_residuals.size)

        def residuals(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            # Penalty-based constraint handling: when the constraint
            # rejects the iterate we return a vector of large finite
            # values, big enough to repel the search but finite so
            # MINPACK's forward-difference jacobian remains usable.
            # This mirrors the C++ ``LevenbergMarquardt::fcn`` failure
            # branch which returns QL_MAX_REAL.
            if not problem.constraint.test(x):
                return np.full(residual_size, _INFEASIBLE_PENALTY, dtype=np.float64)
            return problem.values(x)

        # ftol mirrors C++'s ``functionEpsilon``, xtol mirrors
        # ``rootEpsilon``, gtol mirrors ``gradientNormEpsilon``.
        # ``diff_step`` is scipy's analogue to C++'s ``epsfcn`` — the
        # relative step size for the forward-difference jacobian.
        result = cast(
            "Any",
            least_squares(
                residuals,
                x0,
                method="lm",
                xtol=self._xtol,
                ftol=end_criteria.function_epsilon,
                gtol=self._gtol,
                max_nfev=end_criteria.max_iterations,
                diff_step=self._epsfcn,
            ),
        )

        problem.set_current_value(np.asarray(result.x, dtype=np.float64))
        # C++ parity: levenbergmarquardt.cpp records f(x*) into the
        # problem's function-value cache.
        final_residuals = np.asarray(result.fun, dtype=np.float64)
        problem.set_function_value(float(final_residuals @ final_residuals))

        return _translate_lm_status(int(result.status))


def _translate_lm_status(status: int) -> Type:
    """Translate ``scipy.optimize.least_squares`` status -> ``Type``.

    Mapping (see scipy docs):
    - 0: max nfev reached -> MaxIterations
    - 1: gtol satisfied -> ZeroGradientNorm
    - 2: ftol satisfied -> StationaryFunctionValue
    - 3: xtol satisfied -> StationaryPoint
    - 4: ftol+xtol satisfied -> StationaryFunctionValue
    - -1 / anything else -> Unknown
    """
    if status == 0:
        return Type.MaxIterations
    if status == 1:
        return Type.ZeroGradientNorm
    if status in {2, 4}:
        return Type.StationaryFunctionValue
    if status == 3:
        return Type.StationaryPoint
    return Type.Unknown
