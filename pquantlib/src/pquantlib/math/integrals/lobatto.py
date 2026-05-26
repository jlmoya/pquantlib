"""Gauss-Lobatto adaptive 1-D integration.

# C++ parity: ql/math/integrals/gausslobattointegral.hpp +
# ql/math/integrals/gausslobattointegral.cpp (v1.42.1) class
# ``GaussLobattoIntegral``.

Algorithm reference:
    W. Gander and W. Gautschi, "Adaptive Quadrature - Revisited."
    BIT, 40(1):84-101, March 2000.

The implementation uses a 4-point Gauss-Lobatto rule recursively
subdividing the interval, with an absolute-tolerance check derived
from an a-priori error estimate.

The ``relative_accuracy`` and ``use_convergence_estimate`` parameters
match the C++ constructor; the probe uses the default form
``GaussLobattoIntegral(max_iterations, abs_accuracy)`` which means
``relative_accuracy=None`` (Null<Real>) and ``use_convergence_estimate=True``.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.integrals.integrator import Integrator, RealFunction

# C++ parity: gausslobattointegral.cpp static class constants.
_ALPHA: float = math.sqrt(2.0 / 3.0)
_BETA: float = 1.0 / math.sqrt(5.0)
_X1: float = 0.94288241569547971906
_X2: float = 0.64185334234578130578
_X3: float = 0.23638319966214988028


class GaussLobattoIntegral(Integrator):
    """Adaptive Gauss-Lobatto quadrature (Gander-Gautschi 2000)."""

    def __init__(
        self,
        max_evaluations: int,
        absolute_accuracy: float,
        relative_accuracy: float | None = None,
        use_convergence_estimate: bool = True,
    ) -> None:
        # C++ parity: constructor calls Integrator(absAccuracy, maxIterations).
        super().__init__(absolute_accuracy, max_evaluations)
        self._relative_accuracy: float | None = relative_accuracy
        self._use_convergence_estimate: bool = use_convergence_estimate

    def _integrate(self, f: RealFunction, a: float, b: float) -> float:
        self._set_number_of_evaluations(0)
        calc_abs_tolerance = self._calculate_abs_tolerance(f, a, b)

        self._increase_number_of_evaluations(2)
        return self._adaptive_gauss_lobatto_step(f, a, b, f(a), f(b), calc_abs_tolerance)

    def _calculate_abs_tolerance(self, f: RealFunction, a: float, b: float) -> float:
        rel_tol = max(self._relative_accuracy if self._relative_accuracy is not None else 0.0, QL_EPSILON)

        m = (a + b) / 2
        h = (b - a) / 2
        y1 = f(a)
        y3 = f(m - _ALPHA * h)
        y5 = f(m - _BETA * h)
        y7 = f(m)
        y9 = f(m + _BETA * h)
        y11 = f(m + _ALPHA * h)
        y13 = f(b)

        f1 = f(m - _X1 * h)
        f2 = f(m + _X1 * h)
        f3 = f(m - _X2 * h)
        f4 = f(m + _X2 * h)
        f5 = f(m - _X3 * h)
        f6 = f(m + _X3 * h)

        acc = h * (
            0.0158271919734801831 * (y1 + y13)
            + 0.0942738402188500455 * (f1 + f2)
            + 0.1550719873365853963 * (y3 + y11)
            + 0.1888215739601824544 * (f3 + f4)
            + 0.1997734052268585268 * (y5 + y9)
            + 0.2249264653333395270 * (f5 + f6)
            + 0.2426110719014077338 * y7
        )

        self._increase_number_of_evaluations(13)
        if acc == 0.0 and (f1 != 0.0 or f2 != 0.0 or f3 != 0.0 or f4 != 0.0 or f5 != 0.0 or f6 != 0.0):
            qassert.fail("can not calculate absolute accuracy from relative accuracy")

        r = 1.0
        if self._use_convergence_estimate:
            integral2 = (h / 6) * (y1 + y13 + 5 * (y5 + y9))
            integral1 = (h / 1470) * (77 * (y1 + y13) + 432 * (y3 + y11) + 625 * (y5 + y9) + 672 * y7)

            if abs(integral2 - acc) != 0.0:
                r = abs(integral1 - acc) / abs(integral2 - acc)
            if r == 0.0 or r > 1.0:
                r = 1.0

        if self._relative_accuracy is not None:
            return min(self._absolute_accuracy, acc * rel_tol) / (r * QL_EPSILON)
        return self._absolute_accuracy / (r * QL_EPSILON)

    def _adaptive_gauss_lobatto_step(
        self,
        f: RealFunction,
        a: float,
        b: float,
        fa: float,
        fb: float,
        acc: float,
    ) -> float:
        qassert.require(self._evaluations < self._max_evaluations, "max number of iterations reached")

        h = (b - a) / 2
        m = (a + b) / 2

        mll = m - _ALPHA * h
        ml = m - _BETA * h
        mr = m + _BETA * h
        mrr = m + _ALPHA * h

        fmll = f(mll)
        fml = f(ml)
        fm = f(m)
        fmr = f(mr)
        fmrr = f(mrr)
        self._increase_number_of_evaluations(5)

        integral2 = (h / 6) * (fa + fb + 5 * (fml + fmr))
        integral1 = (h / 1470) * (77 * (fa + fb) + 432 * (fmll + fmrr) + 625 * (fml + fmr) + 672 * fm)

        # avoid 80 bit logic on x86 cpu (C++ comment preserved verbatim).
        dist = acc + (integral1 - integral2)
        if dist == acc or mll <= a or b <= mrr:
            qassert.require(m > a and b > m, "Interval contains no more machine number")
            return integral1
        return (
            self._adaptive_gauss_lobatto_step(f, a, mll, fa, fmll, acc)
            + self._adaptive_gauss_lobatto_step(f, mll, ml, fmll, fml, acc)
            + self._adaptive_gauss_lobatto_step(f, ml, m, fml, fm, acc)
            + self._adaptive_gauss_lobatto_step(f, m, mr, fm, fmr, acc)
            + self._adaptive_gauss_lobatto_step(f, mr, mrr, fmr, fmrr, acc)
            + self._adaptive_gauss_lobatto_step(f, mrr, b, fmrr, fb, acc)
        )
