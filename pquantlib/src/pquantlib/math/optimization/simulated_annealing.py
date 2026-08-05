"""Simulated annealing over a downhill simplex.

# C++ parity: ql/math/optimization/simulatedannealing.hpp (v1.43) —
# header-only template ``SimulatedAnnealing<RNG>``.

*Numerical Recipes in C*, 2nd edition, chapter 10.9 (``amebsa``/``amotsa``),
with the original exit criterion in ``f(x)`` replaced by one on the
simplex size (see :mod:`pquantlib.math.optimization.simplex` for the GSL
reference behind that change).

Every vertex value is perturbed by ``-T * log(u)`` with ``u`` uniform on
(0, 1) before the best/worst ranking, and each trial point is accepted
against the perturbed worst value. The method is therefore stochastic —
but the draws come from QuantLib's own ``MersenneTwisterUniformRng``, so
a faithful port is fully reproducible for a given seed. The generator
defaults to ``MersenneTwisterUniformRng(0)`` exactly as C++'s ``RNG()``
does, i.e. to a clock-seeded stream via ``SeedGenerator`` — pass an
explicitly seeded generator when the run has to be reproducible.

Two cooling schedules, one per C++ constructor:

- ``constant_factor(lambda, T0, epsilon, m, rng)`` — multiply ``T`` by
  ``(1 - epsilon)`` after every ``m`` moves;
- ``constant_budget(lambda, T0, K, alpha, rng)`` — set
  ``T = T0 * (1 - k/K)^alpha`` after every move, so ``T`` reaches zero
  after ``K`` moves and the method degenerates to a deterministic
  simplex thereafter.
"""

from __future__ import annotations

import copy
import math
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib.math.constants import QL_MAX_REAL
from pquantlib.math.optimization.end_criteria import Type
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng

if TYPE_CHECKING:
    from pquantlib.math.optimization.end_criteria import EndCriteria
    from pquantlib.math.optimization.problem import Problem
    from pquantlib.math.randomnumbers.random_number_generator import (
        RandomNumberGenerator,
    )


class Scheme(IntEnum):
    """Cooling schedule.

    # C++ parity: ``SimulatedAnnealing::Scheme`` in
    # ql/math/optimization/simulatedannealing.hpp:50-53 (v1.43).
    """

    ConstantFactor = 0
    ConstantBudget = 1


class SimulatedAnnealing(OptimizationMethod):
    """Simulated annealing over a downhill simplex.

    # C++ parity: ``template <class RNG> class SimulatedAnnealing`` in
    # ql/math/optimization/simulatedannealing.hpp:46-93 (v1.43).

    Prefer the :meth:`constant_factor` / :meth:`constant_budget`
    constructors, which mirror the two C++ constructors; ``__init__``
    takes the union of their parameters because Python has no overloading.
    """

    Scheme = Scheme

    __slots__ = (
        "_alpha",
        "_epsilon",
        "_k",
        "_lambda",
        "_m",
        "_pb",
        "_ptry",
        "_rng",
        "_scheme",
        "_sum",
        "_t0",
        "_values",
        "_vertices",
        "_yb",
        "_ytry",
    )

    def __init__(
        self,
        scheme: Scheme,
        lambda_: float,
        t0: float,
        epsilon: float,
        alpha: float,
        k: int,
        m: int,
        rng: RandomNumberGenerator | None = None,
    ) -> None:
        # C++ parity: simulatedannealing.hpp:58 and :69 — ``const RNG& rng =
        # RNG()``, i.e. a default-constructed Mersenne twister, which seeds
        # itself from the clock-based SeedGenerator.
        if rng is None:
            rng = MersenneTwisterUniformRng(0)
        self._scheme: Scheme = scheme
        self._lambda: float = lambda_
        self._t0: float = t0
        self._epsilon: float = epsilon
        self._alpha: float = alpha
        self._k: int = k
        self._m: int = m
        # C++ stores ``const RNG rng_``, i.e. a COPY of the generator, so
        # two optimizers built from one generator draw independent streams.
        self._rng: RandomNumberGenerator = copy.deepcopy(rng)
        # Per-run state.
        self._vertices: list[npt.NDArray[np.float64]] = []
        self._values: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._sum: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._ptry: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._pb: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        # C++ leaves ``ytry_`` INDETERMINATE until the first amotsa call,
        # yet reads it (``std::isnan(ytry_)``) inside the vertex-initialisation
        # loop at simulatedannealing.hpp:166. Zero is the value a freshly
        # allocated object holds in practice, and it makes that read a
        # no-op; documented rather than silently "fixed".
        self._ytry: float = 0.0
        # Best-ever cost seen by ``amotsa``; ``minimize`` seeds it with
        # QL_MAX_REAL at the start of every run (simulatedannealing.hpp:174).
        self._yb: float = QL_MAX_REAL

    @classmethod
    def constant_factor(
        cls,
        lambda_: float,
        t0: float,
        epsilon: float,
        m: int,
        rng: RandomNumberGenerator | None = None,
    ) -> SimulatedAnnealing:
        """Reduce ``T`` by a factor of ``(1 - epsilon)`` after every ``m`` moves.

        # C++ parity: simulatedannealing.hpp:56-60 — first constructor.
        """
        return cls(Scheme.ConstantFactor, lambda_, t0, epsilon, 0.0, 0, m, rng)

    @classmethod
    def constant_budget(
        cls,
        lambda_: float,
        t0: float,
        k: int,
        alpha: float,
        rng: RandomNumberGenerator | None = None,
    ) -> SimulatedAnnealing:
        """Budget ``K`` moves; ``T = T0 * (1 - k/K)^alpha``, zero past ``K``.

        # C++ parity: simulatedannealing.hpp:68-71 — second constructor.
        """
        return cls(Scheme.ConstantBudget, lambda_, t0, 0.0, alpha, k, 0, rng)

    # --- internals -------------------------------------------------------

    def _simplex_size(self) -> float:
        """Mean Euclidean distance of the vertices from their centroid.

        # C++ parity: simulatedannealing.hpp:96-108 — a verbatim copy of
        # ``computeSimplexSize`` from simplex.cpp.
        """
        center = np.zeros(self._vertices[0].size, dtype=np.float64)
        for vertex in self._vertices:
            center += vertex
        center *= 1.0 / float(len(self._vertices))
        result = 0.0
        for vertex in self._vertices:
            temp = vertex - center
            acc = 0.0
            for i in range(temp.size):
                acc += float(temp[i]) * float(temp[i])
            result += float(np.sqrt(acc))
        return result / float(len(self._vertices))

    def _amotsa(
        self, problem: Problem, fac: float, n: int, ihi: int, tt: float, yhi: float
    ) -> tuple[float, float]:
        """Reflect/expand/contract the (thermally perturbed) worst vertex.

        # C++ parity: simulatedannealing.hpp:111-138 — ``amotsa``.

        C++ communicates through members; the Python port passes ``ihi``,
        ``tt`` and ``yhi`` in and returns ``(ytry, yhi)`` because ``yhi``
        is the only member ``amotsa`` writes that the caller reads back
        before the next assignment.
        """
        fac1 = (1.0 - fac) / float(n)
        fac2 = fac1 - fac
        for j in range(n):
            self._ptry[j] = self._sum[j] * fac1 - self._vertices[ihi][j] * fac2
        if not problem.constraint.test(self._ptry):
            self._ytry = QL_MAX_REAL
        else:
            self._ytry = problem.value(self._ptry)
        if math.isnan(self._ytry):
            self._ytry = QL_MAX_REAL
        if self._ytry <= self._yb:
            self._yb = self._ytry
            self._pb = self._ptry.astype(np.float64, copy=True)
        yflu = self._ytry - tt * math.log(self._rng.next().value)
        if yflu < yhi:
            self._values[ihi] = self._ytry
            yhi = yflu
            for j in range(n):
                self._sum[j] += self._ptry[j] - self._vertices[ihi][j]
                self._vertices[ihi][j] = self._ptry[j]
        self._ytry = yflu
        return self._ytry, yhi

    # --- OptimizationMethod ---------------------------------------------

    def minimize(self, problem: Problem, end_criteria: EndCriteria) -> Type:  # noqa: PLR0915
        # C++ parity: simulatedannealing.hpp:141-273.
        stationary_state_iterations = 0
        ec_type = Type.None_
        problem.reset()
        x = problem.current_value.astype(np.float64, copy=True)
        iteration = 0
        n = int(x.size)
        self._ptry = np.zeros(n, dtype=np.float64)

        # Build the vertices.
        self._vertices = [x.astype(np.float64, copy=True) for _ in range(n + 1)]
        for i in range(n):
            direction = np.zeros(n, dtype=np.float64)
            direction[i] = 1.0
            problem.constraint.update(self._vertices[i + 1], direction, self._lambda)
        self._values = np.zeros(n + 1, dtype=np.float64)
        for i in range(n + 1):
            if not problem.constraint.test(self._vertices[i]):
                self._values[i] = QL_MAX_REAL
            else:
                self._values[i] = problem.value(self._vertices[i])
            # C++ tests ``ytry_``, not ``values_[i_]``, here.
            if math.isnan(self._ytry):
                self._values[i] = QL_MAX_REAL

        # Minimize.
        temperature = self._t0
        self._yb = QL_MAX_REAL
        self._pb = np.zeros(n, dtype=np.float64)
        while True:
            iteration_t = iteration
            while True:
                self._sum = np.zeros(n, dtype=np.float64)
                for i in range(n + 1):
                    self._sum += self._vertices[i]
                tt = -temperature
                ilo = 0
                ihi = 1
                ynhi = float(self._values[0]) + tt * math.log(self._rng.next().value)
                ylo = ynhi
                yhi = float(self._values[1]) + tt * math.log(self._rng.next().value)
                if ylo > yhi:
                    ihi = 0
                    ilo = 1
                    ynhi = yhi
                    yhi = ylo
                    ylo = ynhi
                for i in range(2, n + 1):
                    yt = float(self._values[i]) + tt * math.log(self._rng.next().value)
                    if yt <= ylo:
                        ilo = i
                        ylo = yt
                    if yt > yhi:
                        ynhi = yhi
                        ihi = i
                        yhi = yt
                    elif yt > ynhi:
                        ynhi = yt

                # GSL end criterion in x.
                stationary_state_iterations, hit = end_criteria.check_stationary_point(
                    self._simplex_size(), 0.0, stationary_state_iterations
                )
                if hit is None:
                    hit = end_criteria.check_max_iterations(iteration)
                if hit is not None:
                    ec_type = hit
                    # No matter what, return the best-ever point.
                    problem.set_current_value(self._pb)
                    problem.set_function_value(self._yb)
                    return ec_type

                iteration += 2
                ytry, yhi = self._amotsa(problem, -1.0, n, ihi, tt, yhi)
                if ytry <= ylo:
                    ytry, yhi = self._amotsa(problem, 2.0, n, ihi, tt, yhi)
                elif ytry >= ynhi:
                    ysave = yhi
                    ytry, yhi = self._amotsa(problem, 0.5, n, ihi, tt, yhi)
                    if ytry >= ysave:
                        for i in range(n + 1):
                            if i != ilo:
                                for j in range(n):
                                    self._sum[j] = 0.5 * (
                                        self._vertices[i][j] + self._vertices[ilo][j]
                                    )
                                    self._vertices[i][j] = self._sum[j]
                                # C++ evaluates at ``sum_``, which has just
                                # been overwritten with the new vertex.
                                self._values[i] = problem.value(self._sum)
                        iteration += n
                        for i in range(n):
                            self._sum[i] = 0.0
                        for i in range(n + 1):
                            self._sum += self._vertices[i]
                else:
                    iteration += 1

                if iteration >= iteration_t + (
                    self._m if self._scheme == Scheme.ConstantFactor else 1
                ):
                    break

            if self._scheme == Scheme.ConstantFactor:
                temperature *= 1.0 - self._epsilon
            elif iteration <= self._k:
                temperature = self._t0 * math.pow(
                    1.0 - float(iteration) / float(self._k), self._alpha
                )
            else:
                temperature = 0.0
