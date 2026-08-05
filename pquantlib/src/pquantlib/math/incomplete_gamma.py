"""Incomplete Gamma function.

# C++ parity: ql/math/incompletegamma.{hpp,cpp} (v1.43).

The regularized lower incomplete Gamma function ``P(a, x)``, evaluated by
the Numerical Recipes (2nd ed., ch. 6) pair of representations: the power
series for ``x < a + 1`` and the Lentz continued fraction for ``x >= a + 1``.

Both representations are exported as free functions because the C++ header
exports them; ``CumulativePoissonDistribution`` uses only the dispatching
``incomplete_gamma_function``.

Deliberately NOT delegated to ``scipy.special.gammainc``. scipy's is a
different (Boost/Cephes) evaluation with different convergence behaviour and
a different default accuracy; the C++ answer is defined by *this* series,
this ``accuracy`` (1e-13) and this iteration cap (100) — including the
``QL_FAIL("accuracy not reached")`` when the cap is hit.
"""

from __future__ import annotations

import math
from typing import Final

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.distributions.gamma_function import GammaFunction

_DEFAULT_ACCURACY: Final[float] = 1.0e-13
_DEFAULT_MAX_ITERATION: Final[int] = 100


def incomplete_gamma_function(
    a: float,
    x: float,
    accuracy: float = _DEFAULT_ACCURACY,
    max_iteration: int = _DEFAULT_MAX_ITERATION,
) -> float:
    """Regularized lower incomplete Gamma ``P(a, x)``.

    # C++ parity: ``incompleteGammaFunction`` — incompletegamma.cpp:33-51.
    """
    qassert.require(a > 0.0, "non-positive a is not allowed")
    qassert.require(x >= 0.0, "negative x non allowed")

    if x < (a + 1.0):
        # Series representation.
        return incomplete_gamma_function_series_repr(a, x, accuracy, max_iteration)
    # Continued-fraction representation.
    return 1.0 - incomplete_gamma_function_continued_fraction_repr(a, x, accuracy, max_iteration)


def incomplete_gamma_function_series_repr(
    a: float,
    x: float,
    accuracy: float = _DEFAULT_ACCURACY,
    max_iteration: int = _DEFAULT_MAX_ITERATION,
) -> float:
    """Power-series branch of ``P(a, x)``; valid for ``x < a + 1``.

    # C++ parity: ``incompleteGammaFunctionSeriesRepr`` —
    # incompletegamma.cpp:54-72.
    """
    if x == 0.0:
        return 0.0

    gln = GammaFunction().log_value(a)
    ap = a
    delta = 1.0 / a
    total = delta
    for _ in range(1, max_iteration + 1):
        ap += 1.0
        delta *= x / ap
        total += delta
        if math.fabs(delta) < math.fabs(total) * accuracy:
            return total * math.exp(-x + a * math.log(x) - gln)
    qassert.fail("accuracy not reached")


def incomplete_gamma_function_continued_fraction_repr(
    a: float,
    x: float,
    accuracy: float = _DEFAULT_ACCURACY,
    max_iteration: int = _DEFAULT_MAX_ITERATION,
) -> float:
    """Continued-fraction branch of ``Q(a, x) = 1 - P(a, x)``; ``x >= a + 1``.

    # C++ parity: ``incompleteGammaFunctionContinuedFractionRepr`` —
    # incompletegamma.cpp:74-101.
    """
    gln = GammaFunction().log_value(a)
    b = x + 1.0 - a
    c = 1.0 / QL_EPSILON
    d = 1.0 / b
    h = d
    for i in range(1, max_iteration + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if math.fabs(d) < QL_EPSILON:
            d = QL_EPSILON
        c = b + an / c
        if math.fabs(c) < QL_EPSILON:
            c = QL_EPSILON
        d = 1.0 / d
        delta = d * c
        h *= delta
        if math.fabs(delta - 1.0) < accuracy:
            return math.exp(-x + a * math.log(x) - gln) * h
    qassert.fail("accuracy not reached")


__all__ = [
    "incomplete_gamma_function",
    "incomplete_gamma_function_continued_fraction_repr",
    "incomplete_gamma_function_series_repr",
]
