"""Laguerre-cosine / Laguerre-sine Gaussian quadrature polynomials.

# C++ parity: ql/math/integrals/gausslaguerrecosinepolynomial.hpp (v1.43).

These drive a 1-D quadrature for ``int_0^inf f(x) dx`` whose weight is

* ``w(x; u) = exp(-x) (1 + cos(u x)) / m0``   (cosine flavour), or
* ``w(x; u) = exp(-x) (1 + sin(u x)) / m0``   (sine flavour),

i.e. an exponentially damped trigonometric weight, normalised so the zeroth
moment is one (which :class:`MomentBasedGaussianPolynomial` requires).

The raw moments follow the linear recursion

    m_n = (2 n m_{n-1} - n (n-1) m_{n-2}) / (1 + u^2)

seeded with the two closed forms ``m0()`` / ``m1()`` that differ between the
cosine and sine flavours; ``moment(n)`` then adds ``n!`` (the moment of the
plain ``exp(-x)`` part) and divides by the normalisation.

# C++ parity divergence: the C++ classes are templates ``<mp_real>``; this is
# the ``mp_real == Real`` (double) instantiation only. See
# :mod:`pquantlib.math.integrals.moment_based_gaussian_polynomial`.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from math import isnan, nan

from pquantlib.math.integrals.moment_based_gaussian_polynomial import (
    MomentBasedGaussianPolynomial,
)


class GaussLaguerreTrigonometricBase(MomentBasedGaussianPolynomial):
    """Shared moment recursion for the Laguerre trigonometric polynomials.

    # C++ parity: gausslaguerrecosinepolynomial.hpp:32-77.
    """

    __slots__ = ("_fact_cache", "_moment_cache", "_u")

    def __init__(self, u: float) -> None:
        super().__init__()
        self._u: float = u
        self._moment_cache: list[float] = []
        self._fact_cache: list[float] = []

    @abstractmethod
    def _m0(self) -> float:
        """First seed of the moment recursion (C++ ``m0()``)."""

    @abstractmethod
    def _m1(self) -> float:
        """Second seed of the moment recursion (C++ ``m1()``)."""

    def _raw_moment(self, n: int) -> float:
        """Trigonometric part of the ``n``-th moment.

        # C++ parity: gausslaguerrecosinepolynomial.hpp:41-56 (``moment_``).
        """
        if len(self._moment_cache) <= n:
            self._moment_cache.extend([nan] * (n + 1 - len(self._moment_cache)))

        if isnan(self._moment_cache[n]):
            if n == 0:
                self._moment_cache[0] = self._m0()
            elif n == 1:
                self._moment_cache[1] = self._m1()
            else:
                self._moment_cache[n] = (
                    2 * n * self._raw_moment(n - 1) - n * (n - 1) * self._raw_moment(n - 2)
                ) / (1 + self._u * self._u)

        return self._moment_cache[n]

    def _fact(self, n: int) -> float:
        """``n!`` with the same memoised recursion the C++ uses.

        # C++ parity: gausslaguerrecosinepolynomial.hpp:57-69 (``fact``).
        """
        if len(self._fact_cache) <= n:
            self._fact_cache.extend([nan] * (n + 1 - len(self._fact_cache)))

        if isnan(self._fact_cache[n]):
            if n == 0:
                self._fact_cache[0] = 1.0
            else:
                self._fact_cache[n] = n * self._fact(n - 1)
        return self._fact_cache[n]


class GaussLaguerreCosinePolynomial(GaussLaguerreTrigonometricBase):
    """Gauss-Laguerre-cosine polynomial, weight ``exp(-x)(1+cos(u x))/m0``.

    # C++ parity: gausslaguerrecosinepolynomial.hpp:90-110.
    """

    __slots__ = ("_m0_norm",)

    def __init__(self, u: float) -> None:
        super().__init__(u)
        self._m0_norm: float = 1.0 + 1.0 / (1.0 + u * u)

    def moment(self, i: int) -> float:
        return (self._raw_moment(i) + self._fact(i)) / self._m0_norm

    def w(self, x: float) -> float:
        return math.exp(-x) * (1 + math.cos(self._u * x)) / self._m0_norm

    def _m0(self) -> float:
        return 1 / (1 + self._u * self._u)

    def _m1(self) -> float:
        return (1 - self._u * self._u) / ((1 + self._u * self._u) * (1 + self._u * self._u))


class GaussLaguerreSinePolynomial(GaussLaguerreTrigonometricBase):
    """Gauss-Laguerre-sine polynomial, weight ``exp(-x)(1+sin(u x))/m0``.

    # C++ parity: gausslaguerrecosinepolynomial.hpp:123-143.
    """

    __slots__ = ("_m0_norm",)

    def __init__(self, u: float) -> None:
        super().__init__(u)
        self._m0_norm: float = 1.0 + u / (1.0 + u * u)

    def moment(self, i: int) -> float:
        return (self._raw_moment(i) + self._fact(i)) / self._m0_norm

    def w(self, x: float) -> float:
        return math.exp(-x) * (1 + math.sin(self._u * x)) / self._m0_norm

    def _m0(self) -> float:
        return self._u / (1 + self._u * self._u)

    def _m1(self) -> float:
        return 2 * self._u / ((1 + self._u * self._u) * (1 + self._u * self._u))


__all__ = [
    "GaussLaguerreCosinePolynomial",
    "GaussLaguerreSinePolynomial",
    "GaussLaguerreTrigonometricBase",
]
