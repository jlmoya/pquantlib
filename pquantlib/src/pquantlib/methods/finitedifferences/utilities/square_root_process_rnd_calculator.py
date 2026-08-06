"""SquareRootProcessRNDCalculator — transition density of a CIR/Heston variance.

# C++ parity: ql/methods/finitedifferences/utilities/squarerootprocessrndcalculator.{hpp,cpp}
# (v1.43).

The transition law of ``dv = kappa(theta - v)dt + sigma sqrt(v) dW`` is a
scaled non-central chi-square: with ``d = 4 kappa / sigma^2``,
``df = d theta``, ``e = exp(-kappa t)``, ``k = d/(1-e)`` and
``ncp = k v0 e``, the variable ``k v`` is ``chi^2(df, ncp)``.

The stationary law is a Gamma with shape ``df/2`` and rate ``df/(2 theta)``.

**Delegation, cross-validated.** C++ calls Boost's
``non_central_chi_squared_distribution`` and ``gamma_p``/``gamma_p_inv``
directly, *not* QuantLib's own ``NonCentralCumulativeChiSquareDistribution``
(which is a truncated series with a 1e-12 *absolute* stopping rule and is
therefore useless in the tail — see the note in
``pquantlib.math.distributions.non_central_chi_square_distribution``). The
Python port therefore delegates to ``scipy.stats.ncx2`` and
``scipy.special.gammainc``/``gammaincinv``, which are the same mathematical
objects Boost implements. That delegation is not assumed: every branch here
is pinned against the C++ probe
(``references/v143/methods/rnd.json``), including deep-tail quantiles.
"""

from __future__ import annotations

import math
from typing import final

from scipy.special import (  # pyright: ignore[reportMissingTypeStubs]
    gammainc,  # pyright: ignore[reportUnknownVariableType]
    gammaincinv,  # pyright: ignore[reportUnknownVariableType]
)
from scipy.stats import ncx2  # pyright: ignore[reportMissingTypeStubs]

from pquantlib.methods.finitedifferences.utilities.risk_neutral_density_calculator import (
    RiskNeutralDensityCalculator,
)


@final
class SquareRootProcessRNDCalculator(RiskNeutralDensityCalculator):
    """Transition + stationary density of a square-root (CIR) process.

    # C++ parity: ``class SquareRootProcessRNDCalculator :
    # public RiskNeutralDensityCalculator``.
    """

    __slots__ = ("_d", "_df", "_kappa", "_theta", "_v0")

    def __init__(self, v0: float, kappa: float, theta: float, sigma: float) -> None:
        self._v0: float = v0
        self._kappa: float = kappa
        self._theta: float = theta
        # C++ parity: d_(4*kappa/(sigma*sigma)), df_(d_*theta).
        self._d: float = 4.0 * kappa / (sigma * sigma)
        self._df: float = self._d * theta

    def _params(self, t: float) -> tuple[float, float]:
        """``(k, ncp)`` — the scale and non-centrality at horizon ``t``."""
        e = math.exp(-self._kappa * t)
        k = self._d / (1.0 - e)
        return k, k * self._v0 * e

    def pdf(self, v: float, t: float, /) -> float:
        """# C++ parity: ``pdf`` — ``boost::math::pdf(dist, v*k) * k``."""
        k, ncp = self._params(t)
        return float(ncx2.pdf(v * k, self._df, ncp)) * k  # pyright: ignore[reportUnknownMemberType]

    def cdf(self, v: float, t: float, /) -> float:
        """# C++ parity: ``cdf`` — ``boost::math::cdf(dist, v*k)``."""
        k, ncp = self._params(t)
        return float(ncx2.cdf(v * k, self._df, ncp))  # pyright: ignore[reportUnknownMemberType]

    def invcdf(self, q: float, t: float, /) -> float:
        """# C++ parity: ``invcdf`` — ``boost::math::quantile(dist, q) / k``."""
        k, ncp = self._params(t)
        return float(ncx2.ppf(q, self._df, ncp)) / k  # pyright: ignore[reportUnknownMemberType]

    def stationary_pdf(self, v: float) -> float:
        """# C++ parity: ``stationary_pdf`` — Gamma(alpha=df/2, beta=alpha/theta)."""
        alpha = 0.5 * self._df
        beta = alpha / self._theta
        return math.pow(beta, alpha) * math.pow(v, alpha - 1.0) * math.exp(
            -beta * v - math.lgamma(alpha)
        )

    def stationary_cdf(self, v: float) -> float:
        """# C++ parity: ``stationary_cdf`` — ``boost::math::gamma_p(alpha, beta*v)``."""
        alpha = 0.5 * self._df
        beta = alpha / self._theta
        return float(gammainc(alpha, beta * v))

    def stationary_invcdf(self, q: float) -> float:
        """# C++ parity: ``stationary_invcdf`` — ``gamma_p_inv(alpha, q)/beta``."""
        alpha = 0.5 * self._df
        beta = alpha / self._theta
        return float(gammaincinv(alpha, q)) / beta


__all__ = ["SquareRootProcessRNDCalculator"]
