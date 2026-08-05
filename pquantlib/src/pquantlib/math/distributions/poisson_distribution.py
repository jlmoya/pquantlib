"""Poisson distribution — density, cumulative and inverse cumulative.

# C++ parity: ql/math/distributions/poissondistribution.hpp (v1.43) —
#             header-only; there is no .cpp.

Three classes, all transcribed rather than delegated to ``scipy.stats.poisson``:

* :class:`PoissonDistribution` builds the density from
  ``exp(k*log(mu) - Factorial.ln(k) - mu)`` — i.e. through *QuantLib's*
  ``Factorial``, which is table-driven up to 27 and Lanczos beyond;
* :class:`CumulativePoissonDistribution` is ``1 - P(k+1, mu)`` through
  QuantLib's own :func:`~pquantlib.math.incomplete_gamma.incomplete_gamma_function`;
* :class:`InverseCumulativePoisson` walks the density forward one term at a
  time until the running sum passes ``x`` and returns ``index - 1`` — a
  deliberately naive search whose answer at the boundaries depends on the
  exact summation order.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.math.constants import QL_MAX_REAL
from pquantlib.math.factorial import Factorial
from pquantlib.math.incomplete_gamma import incomplete_gamma_function


class PoissonDistribution:
    """Poisson probability mass at ``k`` for intensity ``mu``.

    # C++ parity: ``class PoissonDistribution`` — poissondistribution.hpp:37-44,
    # inline definitions at :85-102.
    """

    __slots__ = ("_log_mu", "_mu")

    def __init__(self, mu: float) -> None:
        self._mu: float = float(mu)
        qassert.require(self._mu >= 0.0, f"mu must be non negative ({self._mu} not allowed)")
        # C++ leaves logMu_ uninitialised when mu == 0; operator() never reads
        # it on that path, so a NaN-free 0.0 placeholder is equivalent.
        self._log_mu: float = math.log(self._mu) if self._mu != 0.0 else 0.0

    def __call__(self, k: int) -> float:
        # C++ parity: poissondistribution.hpp:92-102.
        if self._mu == 0.0:
            return 1.0 if k == 0 else 0.0
        log_factorial = Factorial.ln(k)
        return math.exp(k * math.log(self._mu) - log_factorial - self._mu)


class CumulativePoissonDistribution:
    """``P[X <= k]`` for ``X ~ Poisson(mu)``.

    # C++ parity: ``class CumulativePoissonDistribution`` —
    # poissondistribution.hpp:56-64.
    """

    __slots__ = ("_mu",)

    def __init__(self, mu: float) -> None:
        self._mu: float = float(mu)

    def __call__(self, k: int) -> float:
        # C++ parity: poissondistribution.hpp:60 — 1 - P(k+1, mu). Note the
        # incomplete-Gamma call, not a closed-form sum: for mu == 0 the x == 0
        # branch of the series returns 0 and the result is exactly 1.
        return 1.0 - incomplete_gamma_function(k + 1, self._mu)


class InverseCumulativePoisson:
    """Smallest ``k`` whose cumulative Poisson mass reaches ``x``.

    # C++ parity: ``class InverseCumulativePoisson`` —
    # poissondistribution.hpp:70-77, inline definitions at :105-131.
    """

    __slots__ = ("_lambda",)

    def __init__(self, lambda_: float = 1.0) -> None:
        self._lambda: float = float(lambda_)
        qassert.require(self._lambda > 0.0, "lambda must be positive")

    def _calc_summand(self, index: int) -> float:
        # C++ parity: poissondistribution.hpp:128-131 — exp(-lambda) *
        # lambda^index / index!, evaluated in exactly this order.
        return math.exp(-self._lambda) * math.pow(self._lambda, index) / Factorial.get(index)

    def __call__(self, x: float) -> float:
        # C++ parity: poissondistribution.hpp:110-126.
        qassert.require(
            0.0 <= x <= 1.0,
            "Inverse cumulative Poisson distribution is only defined on the interval [0,1]",
        )

        if x == 1.0:
            return QL_MAX_REAL

        total = 0.0
        index = 0
        while x > total:
            total += self._calc_summand(index)
            index += 1

        # C++ parity: poissondistribution.hpp:125 — ``return Real(index-1);``
        # with ``index`` a ``BigNatural`` (64-bit *unsigned*). For x == 0 the
        # loop body never runs, index is 0, and ``index-1`` wraps to 2^64-1,
        # so C++ returns 1.8446744073709552e19 rather than -1. The mask
        # reproduces the wraparound; without it the port would silently
        # disagree with C++ on the whole x == 0 boundary.
        return float((index - 1) & 0xFFFF_FFFF_FFFF_FFFF)


__all__ = [
    "CumulativePoissonDistribution",
    "InverseCumulativePoisson",
    "PoissonDistribution",
]
