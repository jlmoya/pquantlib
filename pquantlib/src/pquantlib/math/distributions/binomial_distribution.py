"""Binomial distribution helpers — Peizer-Pratt inversion.

# C++ parity: ql/math/distributions/binomialdistribution.hpp (v1.42.1).

PQuantLib currently ports only the ``PeizerPrattMethod2Inversion`` —
the closed-form approximation to the inverse cumulative binomial that
the Leisen-Reimer + Joshi4 binomial trees use to set up/down probabilities
that converge faster than CRR for European options.

The full ``BinomialDistribution`` class — exact CDF/PDF over an
arbitrary ``(n, p)`` binomial — is a Phase 5 carry-over (no test path
exercises it yet).
"""

from __future__ import annotations

import math

from pquantlib import qassert


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


__all__ = ["peizer_pratt_method2_inversion"]
