"""Binomial distribution — density, cumulative, coefficients, Peizer-Pratt.

# C++ parity: ql/math/distributions/binomialdistribution.hpp (v1.43) —
#             header-only; there is no .cpp.

The density and the binomial coefficients go through QuantLib's own
``Factorial`` (table-driven to 27, Lanczos beyond) rather than
``math.comb``/``math.lgamma``: the C++ answers are those of the Lanczos
approximation, and ``binomial_coefficient`` in particular is
``floor(0.5 + exp(ln C(n,k)))``, so it inherits the Lanczos error before it
is rounded. The cumulative goes through QuantLib's own regularized
incomplete Beta.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.math.beta import incomplete_beta_function
from pquantlib.math.constants import QL_MAX_REAL
from pquantlib.math.factorial import Factorial


def binomial_coefficient_ln(n: int, k: int) -> float:
    """``log C(n, k)`` via ``Factorial.ln``.

    # C++ parity: ``binomialCoefficientLn`` — binomialdistribution.hpp:33-39.
    """
    qassert.require(n >= k, "n<k not allowed")
    return Factorial.ln(n) - Factorial.ln(k) - Factorial.ln(n - k)


def binomial_coefficient(n: int, k: int) -> float:
    """``C(n, k)`` as a float, rounded from the log form.

    # C++ parity: ``binomialCoefficient`` — binomialdistribution.hpp:41-45.
    """
    return math.floor(0.5 + math.exp(binomial_coefficient_ln(n, k)))


class BinomialDistribution:
    """Binomial probability mass at ``k`` for ``n`` trials of probability ``p``.

    # C++ parity: ``class BinomialDistribution`` —
    # binomialdistribution.hpp:51-59, inline definitions at :83-100 and :112-124.
    """

    __slots__ = ("_log_one_minus_p", "_log_p", "_n")

    def __init__(self, p: float, n: int) -> None:
        self._n: int = n
        # C++ parity: binomialdistribution.hpp:83-100. The p == 0 / p == 1
        # sentinels are -QL_MAX_REAL, and operator() dispatches on them by
        # exact equality against 0.0 — so they must be reproduced exactly.
        if p == 0.0:
            self._log_p: float = -QL_MAX_REAL
            self._log_one_minus_p: float = 0.0
        elif p == 1.0:
            self._log_p = 0.0
            self._log_one_minus_p = -QL_MAX_REAL
        else:
            qassert.require(p > 0, "negative p not allowed")
            qassert.require(p < 1.0, "p>1.0 not allowed")
            self._log_p = math.log(p)
            self._log_one_minus_p = math.log(1.0 - p)

    def __call__(self, k: int) -> float:
        # C++ parity: binomialdistribution.hpp:112-124.
        if k > self._n:
            return 0.0
        # p == 1.0
        if self._log_p == 0.0:
            return 1.0 if k == self._n else 0.0
        # p == 0.0
        if self._log_one_minus_p == 0.0:
            return 1.0 if k == 0 else 0.0
        return math.exp(
            binomial_coefficient_ln(self._n, k) + k * self._log_p + (self._n - k) * self._log_one_minus_p
        )


class CumulativeBinomialDistribution:
    """``P[X <= k]`` for ``X ~ Binomial(n, p)``.

    # C++ parity: ``class CumulativeBinomialDistribution`` —
    # binomialdistribution.hpp:66-79, ctor at :102-110.
    """

    __slots__ = ("_n", "_p")

    def __init__(self, p: float, n: int) -> None:
        qassert.require(p >= 0, "negative p not allowed")
        qassert.require(p <= 1.0, "p>1.0 not allowed")
        self._n: int = n
        self._p: float = float(p)

    def __call__(self, k: int) -> float:
        # C++ parity: binomialdistribution.hpp:71-76 — 1 - I_p(k+1, n-k).
        if k >= self._n:
            return 1.0
        return 1.0 - incomplete_beta_function(k + 1, self._n - k, self._p)


def peizer_pratt_method2_inversion(z: float, n: int) -> float:
    """Peizer-Pratt method-2 inversion of the cumulative binomial.

    # C++ parity: ``PeizerPrattMethod2Inversion`` in
    # ql/math/distributions/binomialdistribution.hpp:42-50 (v1.42.1).

    Used by the Leisen-Reimer tree builder. ``n`` must be odd (the
    inversion is only valid for odd ``n``).
    """
    qassert.require(n % 2 == 1, f"n must be an odd number: {n} not allowed")
    result = z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))
    result *= result
    result = math.exp(-result * (n + 1.0 / 6.0))
    sign = 1.0 if z > 0 else -1.0
    return 0.5 + sign * math.sqrt(0.25 * (1.0 - result))


__all__ = [
    "BinomialDistribution",
    "CumulativeBinomialDistribution",
    "binomial_coefficient",
    "binomial_coefficient_ln",
    "peizer_pratt_method2_inversion",
]
