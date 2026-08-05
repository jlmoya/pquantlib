"""Polynomial functional form ``f(t) = sum_i c_i t^i``.

# C++ parity: ql/math/polynomialmathfunction.{hpp,cpp} (v1.43).

Not ``numpy.polynomial.Polynomial``. The value, derivative and primitive are
Horner-free running-power loops whose accumulation order C++ fixes, and the
rolling-window coefficient transforms go through a Pascal-triangle matrix and
its inverse — the inverse in particular is a genuine linear solve whose result
is not recoverable from the polynomial interface.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix
from pquantlib.math.pascal_triangle import PascalTriangle


class PolynomialFunction:
    """``f(t) = sum_{i=0}^{n} c_i t^i``.

    # C++ parity: ``class PolynomialFunction`` —
    # polynomialmathfunction.hpp:31-74, polynomialmathfunction.cpp:23-108.
    """

    __slots__ = ("_c", "_der_c", "_k", "_order", "_pr_c")

    def __init__(self, coeff: Sequence[float]) -> None:
        # C++ parity: polynomialmathfunction.cpp:23-45.
        qassert.require(len(coeff) > 0, "empty coefficient vector")
        self._order: int = len(coeff)
        self._c: list[float] = [float(v) for v in coeff]
        self._der_c: list[float] = [0.0] * (self._order - 1)
        self._pr_c: list[float] = [0.0] * self._order
        self._k: float = 0.0

        i = 0
        for i in range(self._order - 1):
            self._pr_c[i] = self._c[i] / (i + 1)
            self._der_c[i] = self._c[i + 1] * (i + 1)
        if self._order > 1:
            i += 1
        self._pr_c[i] = self._c[i] / (i + 1)

    def __call__(self, t: float) -> float:
        # C++ parity: polynomialmathfunction.cpp:47-54.
        result = 0.0
        t_power = 1.0
        for i in range(self._order):
            result += self._c[i] * t_power
            t_power *= t
        return result

    def derivative(self, t: float) -> float:
        # C++ parity: polynomialmathfunction.cpp:56-63.
        result = 0.0
        t_power = 1.0
        for i in range(self._order - 1):
            result += self._der_c[i] * t_power
            t_power *= t
        return result

    def primitive(self, t: float) -> float:
        # C++ parity: polynomialmathfunction.cpp:65-72.
        result = self._k
        t_power = t
        for i in range(self._order):
            result += self._pr_c[i] * t_power
            t_power *= t
        return result

    def definite_integral(self, t1: float, t2: float) -> float:
        # C++ parity: polynomialmathfunction.cpp:74-77.
        return self.primitive(t2) - self.primitive(t1)

    def order(self) -> int:
        return self._order

    def coefficients(self) -> list[float]:
        return list(self._c)

    def derivative_coefficients(self) -> list[float]:
        return list(self._der_c)

    def primitive_coefficients(self) -> list[float]:
        return list(self._pr_c)

    def _equations(self, t: float, t2: float) -> Matrix:
        """The Pascal-triangle rolling-window matrix.

        # C++ parity: ``initializeEqs_`` — polynomialmathfunction.cpp:79-90.
        Strictly upper-triangular-plus-diagonal; the zero entries below the
        diagonal are what makes the inverse in
        :meth:`definite_derivative_coefficients` well conditioned.
        """
        dt = t2 - t
        eqs = np.zeros((self._order, self._order), dtype=np.float64)
        for i in range(self._order):
            tau = 1.0
            for j in range(i, self._order):
                tau *= dt
                eqs[i, j] = (tau * PascalTriangle.get(j + 1)[i]) / (j + 1)
        return eqs

    def definite_integral_coefficients(self, t: float, t2: float) -> list[float]:
        """Coefficients of the integral over a rolling window ``t2 - t``.

        # C++ parity: polynomialmathfunction.cpp:92-100.
        """
        k = np.asarray(self._c, dtype=np.float64)
        coeff = self._equations(t, t2) @ k
        return [float(v) for v in coeff]

    def definite_derivative_coefficients(self, t: float, t2: float) -> list[float]:
        """Coefficients of the derivative over a rolling window ``t2 - t``.

        # C++ parity: polynomialmathfunction.cpp:102-110. C++ forms
        ``inverse(eqs_) * k`` explicitly; numpy's ``solve`` computes the same
        vector by LU without materialising the inverse, which is the
        numerically better-behaved route to the identical answer for a
        well-conditioned triangular system.
        """
        k = np.asarray(self._c, dtype=np.float64)
        coeff = np.linalg.solve(self._equations(t, t2), k)
        return [float(v) for v in coeff]


__all__ = ["PolynomialFunction"]
