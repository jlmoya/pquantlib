"""ExponentialJump1dMesher — mesher for a fast-mean-reverting jump process.

# C++ parity: ql/methods/finitedifferences/meshers/exponentialjump1dmesher.{hpp,cpp}
# (v1.43).

Grid for

    dY_t = -beta Y_{t-} dt + J_t dN_t,   omega(J) = (1/eta) exp(-J/eta)

Nodes are the inverse of the exponential jump-size distribution on a uniform
probability grid ``p in [0, 1-eps]``, scaled by ``1/(1-exp(-beta/lambda))``.

Reference: B. Hambly, S. Howison, T. Kluge, *Modelling spikes and pricing
swing options in electricity markets*.
"""

from __future__ import annotations

import math
from typing import final

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.distributions.gamma_function import GammaFunction
from pquantlib.math.incomplete_gamma import incomplete_gamma_function
from pquantlib.math.integrals.lobatto import GaussLobattoIntegral
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher


@final
class ExponentialJump1dMesher(Fdm1dMesher):
    """Mesher for an exponential-jump process with high mean reversion.

    # C++ parity: ``class ExponentialJump1dMesher : public Fdm1dMesher``.
    """

    def __init__(
        self,
        steps: int,
        beta: float,
        jump_intensity: float,
        eta: float,
        eps: float = 1e-3,
    ) -> None:
        super().__init__(steps)
        qassert.require(eps > 0.0 and eps < 1.0, "eps > 0.0 and eps < 1.0")
        qassert.require(steps > 1, "minimum number of steps is two")

        self._beta: float = beta
        self._jump_intensity: float = jump_intensity
        self._eta: float = eta

        start = 0.0
        end = 1.0 - eps
        dx = (end - start) / (steps - 1)
        scale = 1.0 / (1.0 - math.exp(-beta / jump_intensity))

        for i in range(steps):
            p = start + i * dx
            self._locations[i] = scale * (-1.0 / eta * math.log(1.0 - p))

        for i in range(steps - 1):
            self._dminus[i + 1] = self._dplus[i] = (
                self._locations[i + 1] - self._locations[i]
            )
        self._dplus[-1] = math.nan
        self._dminus[0] = math.nan

    # --- jump-size distribution (Hambly et al. approximation) -----------

    def jump_size_density_t(self, x: float, t: float) -> float:
        """# C++ parity: ``jumpSizeDensity(Real x, Time t)``."""
        a = 1.0 - self._jump_intensity / self._beta
        norm = 1.0 - math.exp(-self._jump_intensity * t)
        gamma_value = math.exp(GammaFunction().log_value(1.0 - self._jump_intensity / self._beta))
        return (
            self._jump_intensity
            * gamma_value
            / norm
            * (
                incomplete_gamma_function(a, x * math.exp(self._beta * t) * self._eta)
                - incomplete_gamma_function(a, x * self._eta)
            )
            * math.pow(self._eta, self._jump_intensity / self._beta)
            / (self._beta * math.pow(x, a))
        )

    def jump_size_density(self, x: float) -> float:
        """# C++ parity: ``jumpSizeDensity(Real x)`` — the ``t -> inf`` limit."""
        a = 1.0 - self._jump_intensity / self._beta
        gamma_value = math.exp(GammaFunction().log_value(self._jump_intensity / self._beta))
        return (
            math.exp(-x * self._eta)
            * math.pow(x, -a)
            * math.pow(self._eta, 1.0 - a)
            / gamma_value
        )

    def jump_size_distribution_t(self, x: float, t: float) -> float:
        """# C++ parity: ``jumpSizeDistribution(Real x, Time t)``."""
        xmin = min(x, 1.0e-100)
        return GaussLobattoIntegral(1000000, 1e-12)(
            lambda z: self.jump_size_density_t(z, t), xmin, max(x, xmin)
        )

    def jump_size_distribution(self, x: float) -> float:
        """# C++ parity: ``jumpSizeDistribution(Real x)`` — the ``t -> inf`` limit."""
        a = self._jump_intensity / self._beta
        xmin = min(x, QL_EPSILON)
        gamma_value = math.exp(GammaFunction().log_value(self._jump_intensity / self._beta))

        lower_eps = (math.pow(xmin, a) / a - math.pow(xmin, a + 1.0) / (a + 1.0)) / gamma_value

        return lower_eps + GaussLobattoIntegral(10000, 1e-12)(
            self.jump_size_density, xmin / self._eta, max(x, xmin / self._eta)
        )


__all__ = ["ExponentialJump1dMesher"]
