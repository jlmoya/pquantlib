"""Gaussian copula.

# C++ parity: ql/math/copulas/gaussiancopula.{hpp,cpp} (v1.43).

``C(x, y) = Phi_2(Phi^{-1}(x), Phi^{-1}(y); rho)``, built specifically from
``BivariateCumulativeNormalDistributionWe04DP`` and ``InverseCumulativeNormal``
— i.e. West/Genz for the joint and Acklam for the marginal inverse. Both
choices are visible in the answer, so they are named explicitly rather than
reached through the ``BivariateCumulativeNormalDistribution`` typedef.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.math.distributions.bivariate_normal_distribution import (
    BivariateCumulativeNormalDistributionWe04DP,
)
from pquantlib.math.distributions.inverse_cumulative_normal import InverseCumulativeNormal


class GaussianCopula:
    """Gaussian copula with correlation ``rho``.

    # C++ parity: ``class GaussianCopula`` — gaussiancopula.hpp:33-41,
    # gaussiancopula.cpp:24-36.
    """

    __slots__ = ("_bivariate_normal_cdf", "_inv_cum_normal", "_rho")

    def __init__(self, rho: float) -> None:
        qassert.require(rho >= -1.0 and rho <= 1.00, f"rho ({rho}) must be in [-1,1]")
        self._rho: float = float(rho)
        self._bivariate_normal_cdf: BivariateCumulativeNormalDistributionWe04DP = (
            BivariateCumulativeNormalDistributionWe04DP(rho)
        )
        self._inv_cum_normal: InverseCumulativeNormal = InverseCumulativeNormal()

    def __call__(self, x: float, y: float) -> float:
        qassert.require(x >= 0.0 and x <= 1.0, f"1st argument ({x}) must be in [0,1]")
        qassert.require(y >= 0.0 and y <= 1.0, f"2nd argument ({y}) must be in [0,1]")
        return self._bivariate_normal_cdf(self._inv_cum_normal(x), self._inv_cum_normal(y))


__all__ = ["GaussianCopula"]
