"""Beta function and incomplete beta function.

# C++ parity: ql/math/beta.hpp + ql/math/beta.cpp (v1.42.1).

- ``beta_function(z, w)``: closed-form Beta(z, w) = Γ(z)Γ(w)/Γ(z+w),
  computed via ``math.lgamma`` (stdlib, C99-equivalent). C++ uses its own
  ``GammaFunction.logValue`` which we treat as documented-divergence:
  both routes reach the same value at TIGHT tolerance.
- ``incomplete_beta_function(a, b, x, accuracy, max_iteration)``:
  regularized incomplete Beta I_x(a, b) via continued-fraction expansion
  (Numerical Recipes Ch. 6), port of C++ ``incompleteBetaFunction``.
- ``beta_continued_fraction(a, b, x, accuracy, max_iteration)``:
  helper for the continued-fraction expansion; exposed because the C++
  header exports it as a free function.
"""

from __future__ import annotations

import math
from typing import Final

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON

_DEFAULT_ACCURACY: Final[float] = 1e-16
_DEFAULT_MAX_ITER: Final[int] = 100


def beta_function(z: float, w: float) -> float:
    """Mirrors C++ inline ``betaFunction``."""
    return math.exp(math.lgamma(z) + math.lgamma(w) - math.lgamma(z + w))


def beta_continued_fraction(
    a: float,
    b: float,
    x: float,
    accuracy: float = _DEFAULT_ACCURACY,
    max_iteration: int = _DEFAULT_MAX_ITER,
) -> float:
    """Mirrors C++ ``betaContinuedFraction`` — Lentz-style continued fraction."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if math.fabs(d) < QL_EPSILON:
        d = QL_EPSILON
    d = 1.0 / d
    result = d

    for m in range(1, max_iteration + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if math.fabs(d) < QL_EPSILON:
            d = QL_EPSILON
        c = 1.0 + aa / c
        if math.fabs(c) < QL_EPSILON:
            c = QL_EPSILON
        d = 1.0 / d
        result *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if math.fabs(d) < QL_EPSILON:
            d = QL_EPSILON
        c = 1.0 + aa / c
        if math.fabs(c) < QL_EPSILON:
            c = QL_EPSILON
        d = 1.0 / d
        delta = d * c
        result *= delta
        if math.fabs(delta - 1.0) < accuracy:
            return result
    qassert.fail("a or b too big, or maxIteration too small in betacf")


def incomplete_beta_function(
    a: float,
    b: float,
    x: float,
    accuracy: float = _DEFAULT_ACCURACY,
    max_iteration: int = _DEFAULT_MAX_ITER,
) -> float:
    """Mirrors C++ ``incompleteBetaFunction`` — regularized I_x(a, b)."""
    qassert.require(a > 0.0, "a must be greater than zero")
    qassert.require(b > 0.0, "b must be greater than zero")

    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    qassert.require(x > 0.0 and x < 1.0, "x must be in [0,1]")

    result = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return result * beta_continued_fraction(a, b, x, accuracy, max_iteration) / a
    return 1.0 - result * beta_continued_fraction(b, a, 1.0 - x, accuracy, max_iteration) / b
