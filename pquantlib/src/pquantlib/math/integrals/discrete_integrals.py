"""Integrals on non-uniform grids.

# C++ parity: ql/math/integrals/discreteintegrals.{hpp,cpp} (v1.43).

:class:`DiscreteTrapezoidIntegral` and :class:`DiscreteSimpsonIntegral` take a
sampled function ``(x, f)`` — no callable, no refinement — and apply the
composite rule for an arbitrary (possibly non-uniform) abscissa vector.
:class:`DiscreteTrapezoidIntegrator` and :class:`DiscreteSimpsonIntegrator`
are the :class:`~pquantlib.math.integrals.integrator.Integrator` wrappers that
sample a callable on a *uniform* grid of ``maxEvaluations`` points and apply
the same rules with the spacing folded into the weights.

Reference: D. Levy, "Numerical Integration",
http://www2.math.umd.edu/~dlevy/classes/amsc466/lecture-notes/integration-chap.pdf
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.integrals.integrator import Integrator, RealFunction

# C++ ``Null<Real>()`` — ql/utilities/null.hpp yields
# ``std::numeric_limits<float>::max()``.
_NULL_REAL: float = 3.4028234663852886e38


class DiscreteTrapezoidIntegral:
    """Composite trapezoid rule over a sampled, possibly non-uniform grid.

    # C++ parity: discreteintegrals.cpp:24-40.
    """

    __slots__ = ()

    def __call__(self, x: Array, f: Array) -> float:
        n = f.shape[0]
        qassert.require(n == x.shape[0], "inconsistent size")

        if n < 2:
            return 0.0

        total = 0.0
        for i in range(n - 1):
            total += (float(x[i + 1]) - float(x[i])) * (float(f[i]) + float(f[i + 1]))

        return 0.5 * total


class DiscreteSimpsonIntegral:
    """Composite Simpson rule over a sampled, possibly non-uniform grid.

    # C++ parity: discreteintegrals.cpp:42-71. With an even number of samples
    # the last panel has no partner, so a trailing trapezoid term is added.
    """

    __slots__ = ()

    def __call__(self, x: Array, f: Array) -> float:
        n = f.shape[0]
        qassert.require(n == x.shape[0], "inconsistent size")

        if n < 2:
            return 0.0

        total = 0.0

        for j in range(0, max(n - 2, 0), 2):
            dxj = float(x[j + 1]) - float(x[j])
            dxjp1 = float(x[j + 2]) - float(x[j + 1])

            alpha = dxjp1 * (2 * dxj - dxjp1)
            dd = dxj + dxjp1
            k = dd / (6 * dxjp1 * dxj)
            beta = dd * dd
            gamma = dxj * (2 * dxjp1 - dxj)

            total += k * (alpha * float(f[j]) + beta * float(f[j + 1]) + gamma * float(f[j + 2]))

        if (n & 1) == 0:
            total += 0.5 * (float(x[n - 1]) - float(x[n - 2])) * (float(f[n - 1]) + float(f[n - 2]))

        return total


class DiscreteTrapezoidIntegrator(Integrator):
    """Trapezoid rule on a uniform grid of ``evaluations`` points.

    # C++ parity: discreteintegrals.hpp:48-56, discreteintegrals.cpp:73-89.
    """

    __slots__ = ()

    def __init__(self, evaluations: int) -> None:
        # C++ parity: Integrator(Null<Real>(), evaluations).
        super().__init__(_NULL_REAL, evaluations)

    def _integrate(self, f: RealFunction, a: float, b: float) -> float:
        # C++ parity: discreteintegrals.cpp:73-89. The C++ walks the grid by
        # repeatedly adding d to a (accumulated, not a + i*d); kept identical
        # so the sample points round the same way.
        n = self._max_evaluations - 1
        d = (b - a) / n

        total = f(a) * 0.5

        x = a
        for _ in range(n - 1):
            x += d
            total += f(x)

        total += f(b) * 0.5

        self._increase_number_of_evaluations(self._max_evaluations)

        return d * total


class DiscreteSimpsonIntegrator(Integrator):
    """Simpson rule on a uniform grid of ``evaluations`` points.

    # C++ parity: discreteintegrals.hpp:58-66, discreteintegrals.cpp:91-118.
    """

    __slots__ = ()

    def __init__(self, evaluations: int) -> None:
        # C++ parity: Integrator(Null<Real>(), evaluations).
        super().__init__(_NULL_REAL, evaluations)

    def _integrate(self, f: RealFunction, a: float, b: float) -> float:
        # C++ parity: discreteintegrals.cpp:91-118, transcribed literally —
        # including the two successive ``sum *= 2`` (odd-index samples end up
        # with weight 4, even interior ones with weight 2) and the asymmetric
        # 1.5 f(b) + 2.5 f(b-d) closing term taken when n is odd.
        n = self._max_evaluations - 1
        d = (b - a) / n
        d2 = d * 2

        total = 0.0
        x = a + d
        i = 1
        while i < n:
            total += f(x)
            x += d2
            i += 2
        total *= 2

        x = a + d2
        i = 2
        while i < n - 1:
            total += f(x)
            x += d2
            i += 2
        total *= 2

        total += f(a)
        if (n & 1) != 0:
            total += 1.5 * f(b) + 2.5 * f(b - d)
        else:
            total += f(b)

        self._increase_number_of_evaluations(self._max_evaluations)

        return d / 3 * total


__all__ = [
    "DiscreteSimpsonIntegral",
    "DiscreteSimpsonIntegrator",
    "DiscreteTrapezoidIntegral",
    "DiscreteTrapezoidIntegrator",
]
