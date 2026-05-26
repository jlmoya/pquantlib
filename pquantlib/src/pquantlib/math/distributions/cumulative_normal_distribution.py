"""Cumulative normal distribution function.

# C++ parity: ql/math/distributions/normaldistribution.hpp +
#             ql/math/distributions/normaldistribution.cpp (v1.42.1).

The C++ implementation uses ``errorFunction_(z*M_SQRT_2)`` and falls back to
an Abramowitz-Stegun asymptotic expansion (26.2.12) when the principal
result falls below ``1e-8``. The Python port mirrors the same dual-path
algorithm but uses ``math.erf`` (stdlib, C99-equivalent IEEE 754) in place
of the C++ Sun-Microsystems-derived ``ErrorFunction`` approximation.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

from pquantlib import qassert
from pquantlib.math.distributions.normal_distribution import NormalDistribution

# Per ql/mathconstants.hpp the C++ macro ``M_SQRT_2`` is 1/sqrt(2), NOT sqrt(2).
_M_SQRT_2: float = 1.0 / math.sqrt(2.0)
_QL_EPSILON: float = sys.float_info.epsilon
_QL_MAX_REAL: float = sys.float_info.max


@dataclass(frozen=True, slots=True)
class CumulativeNormalDistribution:
    """Cumulative normal CDF at ``x`` with mean ``average`` and stddev ``sigma``."""

    average: float = 0.0
    sigma: float = 1.0
    _gaussian: NormalDistribution = field(
        default_factory=lambda: NormalDistribution(0.0, 1.0),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        qassert.require(self.sigma > 0.0, f"sigma must be greater than 0.0 ({self.sigma} not allowed)")

    def __call__(self, x: float) -> float:
        z = (x - self.average) / self.sigma
        # C++: 0.5 * (1 + erf(z * M_SQRT_2)) where M_SQRT_2 = 1/sqrt(2),
        # i.e. the standard CDF formula Phi(z) = 0.5*(1 + erf(z/sqrt(2))).
        result = 0.5 * (1.0 + math.erf(z * _M_SQRT_2))
        if result <= 1e-8:
            # Abramowitz-Stegun (26.2.12) asymptotic expansion for very
            # negative z. Mirrors C++ loop verbatim.
            zsqr = z * z
            sum_ = 1.0
            i = 1.0
            g = 1.0
            a = _QL_MAX_REAL
            lasta = a
            while True:
                lasta = a
                x_ = (4.0 * i - 3.0) / zsqr
                y_ = x_ * ((4.0 * i - 1.0) / zsqr)
                a = g * (x_ - y_)
                sum_ -= a
                g *= y_
                i += 1.0
                a = math.fabs(a)
                if lasta <= a or a < math.fabs(sum_ * _QL_EPSILON):
                    break
            result = -self._gaussian(z) / z * sum_
        return result

    def derivative(self, x: float) -> float:
        """First derivative: ``gaussian((x - average)/sigma) / sigma``."""
        xn = (x - self.average) / self.sigma
        return self._gaussian(xn) / self.sigma
