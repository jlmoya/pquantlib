"""Downhill-simplex (Nelder-Mead) optimization method.

# C++ parity: ql/math/optimization/simplex.{hpp,cpp} (v1.43).

The algorithm follows *Numerical Recipes in C*, 2nd edition, chapter 10,
with QuantLib's own modifications (simplex.cpp:23-31): the exit criterion
was moved off ``f(x)`` and onto the SIZE OF THE SIMPLEX, following GSL
1.9, because the ``f``-based test reports ``x = 0`` as the minimum of
``x^2 + x + 1`` started from ``x = -100``.

This is a transcription, not a wrapper. It is deliberately NOT
``scipy.optimize.minimize(method="Nelder-Mead")``, and the differences
are not cosmetic:

- **Initial simplex.** Vertex ``i+1`` is built by
  ``Constraint.update(vertex, e_i, lambda)``, i.e. the step is HALVED
  until the vertex is feasible. scipy builds ``x0 + lambda*e_i``
  unconditionally.
- **Stopping rule.** ``computeSimplexSize`` is the mean Euclidean
  distance of the vertices from their centroid, compared against
  ``root_epsilon`` alone. scipy tests ``xatol`` on the max per-coordinate
  spread AND ``fatol`` on the function spread.
- **Constraint handling.** Infeasible trial points are rejected inside
  ``extrapolate`` by halving the reflection ``factor`` until the trial is
  feasible; no penalty value is ever fed to the objective.
- **Two distinct exits.** The simplex-size / max-iteration exit returns
  ``StationaryPoint`` (via the seeded-counter idiom); the
  "cannot extrapolate given the constraints" exit returns
  ``StationaryFunctionValue``. A wrapper that maps a solver status onto
  one of these can never produce the other.

Measured against the C++ probe, the scipy-backed predecessor agreed on
the converged ``x`` to ~1e-12 on well-conditioned problems but differed
in function-evaluation count on 6 of 8 cases, took a completely different
iterate path (max|dx| = 0.70 at a truncated cap of 20 iterations), and
differed by 6.6e-5 in ``x`` once ``root_epsilon`` was loosened to 1e-4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.optimization.end_criteria import Type
from pquantlib.math.optimization.optimization_method import OptimizationMethod

if TYPE_CHECKING:
    from pquantlib.math.optimization.end_criteria import EndCriteria
    from pquantlib.math.optimization.problem import Problem


def _compute_simplex_size(vertices: list[npt.NDArray[np.float64]]) -> float:
    """Mean Euclidean distance of the vertices from their centroid.

    # C++ parity: simplex.cpp:40-51 — the anonymous-namespace
    # ``computeSimplexSize``. Note the centroid is formed by multiplying
    # by the reciprocal (``center *= 1/n``), not by dividing.
    """
    center = np.zeros(vertices[0].size, dtype=np.float64)
    for vertex in vertices:
        center += vertex
    center *= 1.0 / float(len(vertices))
    result = 0.0
    for vertex in vertices:
        temp = vertex - center
        # Norm2 == sqrt(DotProduct(v, v)).
        acc = 0.0
        for i in range(temp.size):
            acc += float(temp[i]) * float(temp[i])
        result += float(np.sqrt(acc))
    return result / float(len(vertices))


class Simplex(OptimizationMethod):
    """Multi-dimensional downhill simplex.

    # C++ parity: ``class Simplex`` in ql/math/optimization/simplex.hpp:58-72
    # (v1.43).

    ``lambda_`` is the characteristic length scale of the problem: the
    initial simplex is ``x0`` plus ``lambda * e_i`` for each coordinate.
    C++ has no default for it (``Simplex(Real lambda)``); the Python port
    keeps the historical ``lambda_ = 1.0`` default for backwards
    compatibility with existing pquantlib callers.
    """

    __slots__ = ("_lambda", "_sum", "_values", "_vertices")

    def __init__(self, lambda_: float = 1.0) -> None:
        # C++ parity: simplex.hpp:61.
        self._lambda: float = lambda_
        self._vertices: list[npt.NDArray[np.float64]] = []
        self._values: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._sum: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)

    @property
    def lambda_(self) -> float:
        """Characteristic edge length of the initial simplex.

        # C++ parity: simplex.hpp:63 — ``Real lambda() const``.
        """
        return self._lambda

    def _extrapolate(
        self, problem: Problem, i_highest: int, factor: float
    ) -> tuple[float, float]:
        """Reflect/expand/contract the worst vertex through the opposite face.

        # C++ parity: simplex.cpp:54-78 — ``Simplex::extrapolate``.

        C++ takes ``Real& factor`` as an in-out parameter; Python returns
        ``(v_try, factor)``. When the trial point cannot be made feasible
        the loop drives ``factor`` to zero and the CURRENT value at
        ``i_highest`` is returned untouched — the caller detects that via
        ``abs(factor) <= QL_EPSILON``.
        """
        p_try = np.empty(0, dtype=np.float64)
        while True:
            dimensions = self._values.size - 1
            factor1 = (1.0 - factor) / dimensions
            factor2 = factor1 - factor
            p_try = self._sum * factor1 - self._vertices[i_highest] * factor2
            factor *= 0.5
            if problem.constraint.test(p_try) or abs(factor) <= QL_EPSILON:
                break
        if abs(factor) <= QL_EPSILON:
            return float(self._values[i_highest]), factor
        factor *= 2.0
        v_try = problem.value(p_try)
        if v_try < self._values[i_highest]:
            self._values[i_highest] = v_try
            self._sum += p_try - self._vertices[i_highest]
            self._vertices[i_highest] = p_try
        return v_try, factor

    def minimize(self, problem: Problem, end_criteria: EndCriteria) -> Type:  # noqa: PLR0915
        # C++ parity: simplex.cpp:81-192.
        # End criteria on x, per GSL 1.9 (simplex.cpp:85). The commented-out
        # ``ftol`` end criterion on f(x) is NOT used.
        xtol = end_criteria.root_epsilon
        max_stationary_state_iterations = end_criteria.max_stationary_state
        ec_type = Type.None_
        problem.reset()

        x = problem.current_value.astype(np.float64, copy=True)
        qassert.require(
            problem.constraint.test(x),
            f"Initial guess {x} is not in the feasible region.",
        )

        iteration_number = 0

        # Initialize the vertices of the simplex.
        n = int(x.size)
        self._vertices = [x.astype(np.float64, copy=True) for _ in range(n + 1)]
        for i in range(n):
            direction = np.zeros(n, dtype=np.float64)
            direction[i] = 1.0
            # Constraint.update HALVES lambda until the vertex is feasible.
            problem.constraint.update(self._vertices[i + 1], direction, self._lambda)
        # Initialize the function values at the vertices.
        self._values = np.zeros(n + 1, dtype=np.float64)
        for i in range(n + 1):
            self._values[i] = problem.value(self._vertices[i])

        # Loop looking for the minimum.
        while True:
            self._sum = np.zeros(n, dtype=np.float64)
            for i in range(n + 1):
                self._sum += self._vertices[i]
            # Determine the best (i_lowest), worst (i_highest) and
            # second-worst (i_next_highest) vertices.
            i_lowest = 0
            if self._values[0] < self._values[1]:
                i_highest = 1
                i_next_highest = 0
            else:
                i_highest = 0
                i_next_highest = 1
            # C++ parity: simplex.cpp:126 — the scan starts at i = 1, so
            # vertex 1 is examined a second time. Preserved deliberately.
            for i in range(1, n + 1):
                if self._values[i] > self._values[i_highest]:
                    i_next_highest = i_highest
                    i_highest = i
                elif self._values[i] > self._values[i_next_highest] and i != i_highest:
                    i_next_highest = i
                if self._values[i] < self._values[i_lowest]:
                    i_lowest = i

            # GSL exit strategy on x.
            simplex_size = _compute_simplex_size(self._vertices)
            iteration_number += 1
            if simplex_size < xtol or (
                end_criteria.check_max_iterations(iteration_number) is not None
            ):
                # Seeding the counter with the maximum makes this fire
                # unconditionally -> StationaryPoint.
                _, hit = end_criteria.check_stationary_point(
                    0.0, 0.0, max_stationary_state_iterations
                )
                if hit is not None:
                    ec_type = hit
                hit = end_criteria.check_max_iterations(iteration_number)
                if hit is not None:
                    ec_type = hit
                x = self._vertices[i_lowest]
                low = float(self._values[i_lowest])
                problem.set_function_value(low)
                problem.set_current_value(x)
                return ec_type

            # If the end criteria are not met, continue.
            factor = -1.0
            v_try, factor = self._extrapolate(problem, i_highest, factor)
            if v_try <= self._values[i_lowest] and factor == -1.0:
                factor = 2.0
                _, factor = self._extrapolate(problem, i_highest, factor)
            elif abs(factor) > QL_EPSILON:
                if v_try >= self._values[i_next_highest]:
                    v_save = float(self._values[i_highest])
                    factor = 0.5
                    v_try, factor = self._extrapolate(problem, i_highest, factor)
                    if v_try >= v_save and abs(factor) > QL_EPSILON:
                        for i in range(n + 1):
                            if i != i_lowest:
                                self._vertices[i] = 0.5 * (
                                    self._vertices[i] + self._vertices[i_lowest]
                                )
                                self._values[i] = problem.value(self._vertices[i])

            # If we cannot extrapolate given the constraints, exit.
            if abs(factor) <= QL_EPSILON:
                x = self._vertices[i_lowest]
                low = float(self._values[i_lowest])
                problem.set_function_value(low)
                problem.set_current_value(x)
                return Type.StationaryFunctionValue
