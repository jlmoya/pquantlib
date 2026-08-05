"""Student's t-distribution — density, cumulative and inverse cumulative.

# C++ parity: ql/math/distributions/studenttdistribution.{hpp,cpp} (v1.43).

Not ``scipy.stats.t``. The C++ density is built from *QuantLib's* Lanczos
``GammaFunction.logValue`` and the C++ CDF from *QuantLib's* continued-fraction
regularized incomplete Beta; both carry the accuracy of those approximations,
which is not the accuracy of scipy's. The inverse is a bare Newton iteration
started at ``x = 0`` with a 1e-6 accuracy target and a 50-step cap — a
different root, in general, from a converged quantile, and it raises when the
cap is hit.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.math.beta import incomplete_beta_function
from pquantlib.math.distributions.gamma_function import GammaFunction


class StudentDistribution:
    """Student-t probability density with ``n`` degrees of freedom.

    # C++ parity: ``class StudentDistribution`` —
    # studenttdistribution.hpp:41-49, studenttdistribution.cpp:24-32.
    """

    __slots__ = ("_n",)

    def __init__(self, n: int) -> None:
        qassert.require(n > 0, "invalid parameter for t-distribution")
        self._n: int = n

    def __call__(self, x: float) -> float:
        # C++ parity: studenttdistribution.cpp:25-31 — evaluated in exactly
        # this grouping: g1 / (g2 * power * sqrt(pi * n)).
        g = GammaFunction()
        n = self._n
        g1 = math.exp(g.log_value(0.5 * (n + 1)))
        g2 = math.exp(g.log_value(0.5 * n))

        power = math.pow(1.0 + x * x / n, 0.5 * (n + 1))

        return g1 / (g2 * power * math.sqrt(math.pi * n))


class CumulativeStudentDistribution:
    """Student-t cumulative distribution with ``n`` degrees of freedom.

    # C++ parity: ``class CumulativeStudentDistribution`` —
    # studenttdistribution.hpp:64-72, studenttdistribution.cpp:34-40.
    """

    __slots__ = ("_n",)

    def __init__(self, n: int) -> None:
        qassert.require(n > 0, "invalid parameter for t-distribution")
        self._n: int = n

    def __call__(self, x: float) -> float:
        # C++ parity: studenttdistribution.cpp:35-39. The difference
        # ``I(1; n/2, 1/2) - I(xx; n/2, 1/2)`` is written out rather than
        # simplified to ``1 - I(xx)``: the first term is not exactly 1 in
        # floating point and the port must not "clean it up".
        n = self._n
        xx = 1.0 * n / (x * x + n)
        sig = 1.0 if x > 0 else -1.0

        return 0.5 + 0.5 * sig * (
            incomplete_beta_function(0.5 * n, 0.5, 1.0) - incomplete_beta_function(0.5 * n, 0.5, xx)
        )


class InverseCumulativeStudent:
    """Newton inversion of :class:`CumulativeStudentDistribution`.

    # C++ parity: ``class InverseCumulativeStudent`` —
    # studenttdistribution.hpp:79-90, studenttdistribution.cpp:42-60.

    The iteration always starts at ``x = 0``, takes at least one step, and
    raises once ``max_iterations`` is exhausted without ``|F(x) - y|`` falling
    under ``accuracy``. Both the start point and the cap are part of the
    answer; do not substitute a bracketing solver.
    """

    __slots__ = ("_accuracy", "_d", "_f", "_max_iterations")

    def __init__(self, n: int, accuracy: float = 1e-6, max_iterations: int = 50) -> None:
        self._d: StudentDistribution = StudentDistribution(n)
        self._f: CumulativeStudentDistribution = CumulativeStudentDistribution(n)
        self._accuracy: float = accuracy
        self._max_iterations: int = max_iterations

    def __call__(self, y: float) -> float:
        # C++ parity: studenttdistribution.cpp:43-59.
        qassert.require(0 <= y <= 1, "argument out of range [0, 1]")

        x = 0.0
        count = 0

        # do a few newton steps to find x
        while True:
            x -= (self._f(x) - y) / self._d(x)
            count += 1
            if not (math.fabs(self._f(x) - y) > self._accuracy and count < self._max_iterations):
                break

        qassert.require(
            count < self._max_iterations,
            f"maximum number of iterations {self._max_iterations} reached in "
            f"InverseCumulativeStudent, y={y}, x={x}",
        )

        return x


__all__ = [
    "CumulativeStudentDistribution",
    "InverseCumulativeStudent",
    "StudentDistribution",
]
