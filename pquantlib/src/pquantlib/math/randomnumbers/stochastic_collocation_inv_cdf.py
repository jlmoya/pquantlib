"""Stochastic collocation inverse cumulative distribution function.

# C++ parity: ql/math/randomnumbers/stochasticcollocationinvcdf.{hpp,cpp}
#             (v1.43) — ``class StochasticCollocationInvCDF``.

Grzelak, Witteveen, Suarez-Taboada and Oosterlee, "The Stochastic Collocation
Monte Carlo Sampler: Highly efficient sampling from expensive distributions"
(SSRN 2529691).

The idea: sampling an expensive distribution by inverting its CDF once per
draw is prohibitive, so invert it at a handful of *collocation points* — the
Gauss-Hermite nodes scaled by sqrt(2), which are where a standard normal puts
its mass — and interpolate in between with a Lagrange polynomial. Drawing then
costs one cheap normal inverse plus one polynomial evaluation.

``sigma`` rescales the normal argument so that the outermost collocation node
lands on a caller-chosen tail probability: pass ``p_max`` to pin the right
node, ``p_min`` to pin the left one. Passing neither leaves ``sigma = 1``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final

import numpy as np

from pquantlib.math.array import Array
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.integrals.gaussian_quadrature import GaussHermiteIntegration
from pquantlib.math.interpolations.lagrange_interpolation import LagrangeInterpolation

#: # C++ parity: ``M_SQRT2`` from ql/mathconstants.hpp.
_M_SQRT2: Final[float] = math.sqrt(2.0)


def _g(sigma: float, x: Array, inv_cdf: Callable[[float], float]) -> Array:
    """Collocation ordinates ``y_i = invCDF(N(x_i / sigma))``.

    # C++ parity: the anonymous-namespace ``g``
    # (stochasticcollocationinvcdf.cpp:31-43).
    """
    normal_cdf = CumulativeNormalDistribution()
    y = np.zeros(x.shape[0], dtype=np.float64)
    for i in range(x.shape[0]):
        y[i] = inv_cdf(normal_cdf(float(x[i]) / sigma))
    return y


class StochasticCollocationInvCDF:
    """Lagrange-collocation approximation of an inverse CDF.

    # C++ parity: ``StochasticCollocationInvCDF``
    # (stochasticcollocationinvcdf.hpp:42-58, .cpp:45-64).

    Args:
        inv_cdf: the expensive inverse cumulative distribution to approximate.
        lagrange_order: number of collocation points (Gauss-Hermite order).
        p_max: if given, ``sigma`` is chosen so the largest collocation node
            maps to this probability.
        p_min: if given (and ``p_max`` is not), likewise for the smallest
            node.
    """

    __slots__ = ("_interpl", "_sigma", "_x", "_y")

    def __init__(
        self,
        inv_cdf: Callable[[float], float],
        lagrange_order: int,
        p_max: float | None = None,
        p_min: float | None = None,
    ) -> None:
        # C++ parity: stochasticcollocationinvcdf.cpp:45-58. Member
        # initialisation order matters — sigma_ reads x_, and y_ reads sigma_.
        self._x: Array = _M_SQRT2 * GaussHermiteIntegration(lagrange_order).x()
        icn = InverseCumulativeNormal()
        if p_max is not None:
            self._sigma: float = float(self._x[-1]) / icn(p_max)
        elif p_min is not None:
            self._sigma = float(self._x[0]) / icn(p_min)
        else:
            self._sigma = 1.0
        self._y: Array = _g(self._sigma, self._x, inv_cdf)
        self._interpl: LagrangeInterpolation = LagrangeInterpolation(self._x, self._y)

    def value(self, x: float) -> float:
        """Evaluate at a *normal* argument ``x``.

        # C++ parity: ``value`` (stochasticcollocationinvcdf.cpp:60-62) —
        # extrapolation is allowed, which is the point: the collocation nodes
        # only span a few standard deviations.
        """
        return self._interpl(x * self._sigma, allow_extrapolation=True)

    def __call__(self, u: float) -> float:
        """Evaluate at a *probability* ``u``.

        # C++ parity: ``operator()`` (stochasticcollocationinvcdf.cpp:63-65).
        """
        return self.value(InverseCumulativeNormal()(u))

    def x(self) -> Array:
        """The collocation abscissae ``sqrt(2) * GaussHermite nodes``."""
        return self._x.copy()

    def y(self) -> Array:
        """The collocation ordinates."""
        return self._y.copy()

    def sigma(self) -> float:
        """The normal-argument rescaling implied by ``p_max`` / ``p_min``."""
        return self._sigma


__all__ = ["StochasticCollocationInvCDF"]
