"""Normal probability-density function.

# C++ parity: ql/math/distributions/normaldistribution.hpp +
#             ql/math/distributions/normaldistribution.cpp (v1.42.1).

The C++ class precomputes ``normalizationFactor_ = sqrt(2)/sqrt(pi)/sigma``
and ``denominator_ = 2 * sigma^2`` in the ctor and reuses them in
``operator()``. The Python port mirrors that with a frozen dataclass and
derived ``__post_init__`` caches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pquantlib import qassert


@dataclass(frozen=True, slots=True)
class NormalDistribution:
    """Normal pdf at ``x`` with mean ``average`` and stddev ``sigma``."""

    average: float = 0.0
    sigma: float = 1.0
    _normalization_factor: float = field(init=False, repr=False, compare=False)
    _denominator: float = field(init=False, repr=False, compare=False)
    _der_normalization_factor: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        qassert.require(self.sigma > 0.0, f"sigma must be greater than 0.0 ({self.sigma} not allowed)")
        # Mirrors C++ ctor: normalizationFactor_ = M_SQRT_2 * M_1_SQRTPI / sigma_
        # where (per ql/mathconstants.hpp) M_SQRT_2 = 1/sqrt(2) and
        # M_1_SQRTPI = 1/sqrt(pi). Equivalent to 1/(sigma * sqrt(2*pi)).
        norm = (1.0 / math.sqrt(2.0)) * (1.0 / math.sqrt(math.pi)) / self.sigma
        der_norm = self.sigma * self.sigma
        object.__setattr__(self, "_normalization_factor", norm)
        object.__setattr__(self, "_der_normalization_factor", der_norm)
        object.__setattr__(self, "_denominator", 2.0 * der_norm)

    def __call__(self, x: float) -> float:
        deltax = x - self.average
        exponent = -(deltax * deltax) / self._denominator
        # C++ guards against very negative exponents to dodge denormal slowdown
        # on a now-extinct Debian Alpha box. Preserved for parity.
        if exponent <= -690.0:
            return 0.0
        return self._normalization_factor * math.exp(exponent)

    def derivative(self, x: float) -> float:
        """First derivative of the pdf at ``x``.

        Mirrors C++ ``derivative``: ``(*this)(x) * (average - x) / sigma^2``.
        """
        return (self(x) * (self.average - x)) / self._der_normalization_factor


# Backward-compat alias used in C++ headers (``typedef NormalDistribution
# GaussianDistribution``).
GaussianDistribution = NormalDistribution
