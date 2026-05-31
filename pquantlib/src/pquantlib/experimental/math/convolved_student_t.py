"""ConvolvedStudentT — cumulative of a linear combination of Student-t variables.

# C++ parity: ql/experimental/math/convolvedstudentt.{hpp,cpp} @ v1.42.1 (099987f0).

Exact analytical computation of the cumulative distribution of a linear
combination of an arbitrary number of independent Student-t variables of **odd**
integer order (the generalised Behrens-Fisher distribution). Adapted from the
algorithm in V. Witkovsky, *Journal of Statistical Planning and Inference* 94
(2001) 1-13.

Two classes are provided, matching the C++ names:

  * :class:`CumulativeBehrensFisher` — the forward CDF / density via the
    Gil-Pelaez inversion theorem applied analytically to the product of the
    individual characteristic functions (each a polynomial times a decaying
    exponential).
  * :class:`InverseCumulativeBehrensFisher` — the quantile, found with a Brent
    root solve over the bracket implied by the normal upper bound.

The public alias ``ConvolvedStudentT`` is the cumulative distribution (the name
used in the W6-C task surface). Only odd degrees of freedom are supported, as in
C++.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.factorial import Factorial
from pquantlib.math.solvers1d.brent import Brent


class CumulativeBehrensFisher:
    """Cumulative (generalised) Behrens-Fisher distribution (odd orders only).

    :param degrees_freedom: degrees of freedom of the convolved Ts (odd, >= 0).
    :param factors: factors in the linear combination of the Ts.
    """

    __slots__ = (
        "_a",
        "_a2",
        "_degrees_freedom",
        "_factors",
        "_poly_convolved",
    )

    def __init__(
        self,
        degrees_freedom: Sequence[int] | None = None,
        factors: Sequence[float] | None = None,
    ) -> None:
        df = list(degrees_freedom) if degrees_freedom is not None else []
        fac = list(factors) if factors is not None else []
        qassert.require(
            len(df) == len(fac), "Incompatible sizes in convolution."
        )
        for i in df:
            qassert.require(i % 2 != 0, "Even degree of freedom not allowed")
            qassert.require(i >= 0, "Negative degree of freedom not allowed")

        self._degrees_freedom = df
        self._factors = fac

        poly_char_fnc: list[list[float]] = [
            self._polyn_charact_t((d - 1) // 2) for d in df
        ]
        # adjust polynomial coefficients by the linear-combination factors:
        for i in range(len(df)):
            multiplier = 1.0
            for k in range(1, len(poly_char_fnc[i])):
                multiplier *= abs(fac[i])
                poly_char_fnc[i][k] *= multiplier

        # convolution = product of the polynomials:
        poly_convolved: list[float] = [1.0]
        for poly in poly_char_fnc:
            poly_convolved = self._convolve_vector_polynomials(
                poly_convolved, poly
            )
        # trim trailing zeros:
        while poly_convolved and poly_convolved[-1] == 0.0:
            poly_convolved.pop()
        self._poly_convolved = poly_convolved

        # cache 'a' (the exponential exponent) and its square:
        a = 0.0
        for i in range(len(df)):
            a += math.sqrt(float(df[i])) * abs(fac[i])
        self._a = a
        self._a2 = a * a

    def degree_freedom(self) -> list[int]:
        return self._degrees_freedom

    def factors(self) -> list[float]:
        return self._factors

    # ---- characteristic-function polynomials ----

    @staticmethod
    def _polyn_charact_t(n: int) -> list[float]:
        """Coefficients of the characteristic-function polynomial of T_(2n+1)."""
        nu = 2 * n + 1
        low: list[float] = [1.0]
        high: list[float] = [1.0, math.sqrt(float(nu))]
        if n == 0:
            return low
        if n == 1:
            return high
        for k in range(1, n):
            recursion_factor = [0.0, 0.0, nu / ((2.0 * k + 1.0) * (2.0 * k - 1.0))]
            low_up = CumulativeBehrensFisher._convolve_vector_polynomials(
                recursion_factor, low
            )
            for i in range(len(high)):
                low_up[i] += high[i]
            low = high
            high = low_up
        return high

    @staticmethod
    def _convolve_vector_polynomials(
        v1: Sequence[float], v2: Sequence[float]
    ) -> list[float]:
        """Polynomial product (convolution of coefficient vectors)."""
        shorter = v1 if len(v1) < len(v2) else v2
        longer = v2 if shorter is v1 else v1
        new_degree = len(v1) + len(v2) - 2
        result = [0.0] * (new_degree + 1)
        for poly_order in range(len(result)):
            lo = max(0, poly_order - len(longer) + 1)
            hi = min(poly_order, len(shorter) - 1)
            for i in range(lo, hi + 1):
                result[poly_order] += shorter[i] * longer[poly_order - i]
        return result

    # ---- cumulative / density via Gil-Pelaez ----

    def __call__(self, x: float) -> float:
        """Cumulative probability ``P(sum a_i T_i <= x)``."""
        pc = self._poly_convolved
        integral = pc[0] * math.atan(x / self._a)
        squared = self._a2 + x * x
        rootsqr = math.sqrt(squared)
        atan2xa = math.atan2(-x, self._a)
        if len(pc) > 1:
            integral += pc[1] * x / squared
        for exponent in range(2, len(pc)):
            integral -= (
                pc[exponent]
                * Factorial.get(exponent - 1)
                * math.sin(exponent * atan2xa)
                / rootsqr**float(exponent)
            )
        return 0.5 + integral / math.pi

    def density(self, x: float) -> float:
        """Probability density of the convolved distribution at ``x``."""
        pc = self._poly_convolved
        squared = self._a2 + x * x
        integral = pc[0] * self._a / squared
        rootsqr = math.sqrt(squared)
        atan2xa = math.atan2(-x, self._a)
        for exponent in range(1, len(pc)):
            integral += (
                pc[exponent]
                * Factorial.get(exponent)
                * math.cos((exponent + 1) * atan2xa)
                / rootsqr ** float(exponent + 1)
            )
        return integral / math.pi


class InverseCumulativeBehrensFisher:
    """Inverse of the convolved odd-T cumulative, via a Brent root solve.

    :param degrees_freedom: degrees of freedom of the convolved Ts (odd).
    :param factors: factors in the linear combination of the Ts.
    :param accuracy: accuracy of the root-solving process.
    """

    __slots__ = ("_accuracy", "_distrib", "_norm_sqr")

    def __init__(
        self,
        degrees_freedom: Sequence[int] | None = None,
        factors: Sequence[float] | None = None,
        accuracy: float = 1.0e-6,
    ) -> None:
        fac = list(factors) if factors is not None else []
        self._norm_sqr = sum(f * f for f in fac)
        self._accuracy = accuracy
        self._distrib = CumulativeBehrensFisher(degrees_freedom, factors)

    def __call__(self, q: float) -> float:
        """Quantile ``x`` such that ``CDF(x) == q``."""
        if q == 0.5:
            return 0.0
        if q < 0.5:
            sign = -1.0
            effective_q = 1.0 - q
        else:
            sign = 1.0
            effective_q = q
        x_min = InverseCumulativeNormal.standard_value(effective_q) * self._norm_sqr
        x_max = 1.0e6
        root = Brent().solve(
            lambda x: self._distrib(x) - effective_q,
            self._accuracy,
            (x_min + x_max) / 2.0,
            x_min,
            x_max,
        )
        return sign * root


# Public alias used by the W6-C task surface: ``ConvolvedStudentT`` is the
# cumulative distribution of the sum of n iid odd-order Student-t variables.
ConvolvedStudentT = CumulativeBehrensFisher
