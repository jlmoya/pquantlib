"""Bivariate cumulative normal distribution.

# C++ parity: ql/math/distributions/bivariatenormaldistribution.{hpp,cpp}
# (v1.42.1) — exposes both ``BivariateCumulativeNormalDistributionDr78``
# (Drezner 1978, 6dp) and the default-typedef
# ``BivariateCumulativeNormalDistribution`` (West 2004 / Genz 2004,
# double precision).

The Python port wraps ``scipy.stats.multivariate_normal.cdf`` for the
2D case. This is **at-or-above** the precision of either C++ variant
(Genz 2004 implementation), so we expose a single
``BivariateCumulativeNormalDistribution`` class that matches the C++
``BivariateCumulativeNormalDistributionWe04DP`` precision and serves
as the default. The Dr78 alias is kept for callers that explicitly
want the lower-precision variant — they get the same scipy
implementation.
"""

from __future__ import annotations

from typing import Any, Final, cast

import numpy as np
from scipy.stats import (  # pyright: ignore[reportMissingTypeStubs]
    multivariate_normal as _mvn,
)

from pquantlib import qassert
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)

_MEAN_ZERO: Final[list[float]] = [0.0, 0.0]
_CND: Final[CumulativeNormalDistribution] = CumulativeNormalDistribution()


class BivariateCumulativeNormalDistribution:
    """Bivariate cumulative standard normal CDF: P(X <= a, Y <= b).

    # C++ parity:
    # ``BivariateCumulativeNormalDistributionWe04DP`` (West 2004 /
    # Genz 2004 hybrid numerical integration). The Python port
    # delegates to ``scipy.stats.multivariate_normal.cdf`` whose
    # Genz-Bretz algorithm provides equivalent accuracy.
    """

    def __init__(self, rho: float) -> None:
        qassert.require(rho >= -1.0, f"rho must be >= -1.0 ({rho} not allowed)")
        qassert.require(rho <= 1.0, f"rho must be <= 1.0 ({rho} not allowed)")
        self._rho: float = rho
        self._cov: np.ndarray = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)

    def __call__(self, a: float, b: float) -> float:
        """Return P(X <= a, Y <= b) for standard bivariate normal."""
        # |rho| == 1 is admitted by the constructor above and is reached
        # by real engines (both partial-time lookback engines build the
        # degenerate copula when the lookback window coincides with the
        # option's own window). The covariance matrix is singular there,
        # and scipy's ``cdf`` refuses it outright with
        # ``LinAlgError: the input matrix must be symmetric positive
        # definite``, so the two degenerate limits are taken in closed
        # form. This is not an approximation of C++: at |rho| == 1 the
        # Genz series block is skipped entirely
        # (bivariatenormaldistribution.cpp:212 ``if (fabs(correlation_) < 1)``)
        # and ONLY the closing correction survives, which is exactly the
        # comonotone / countermonotone limit computed below.
        if self._rho == 1.0:
            # cpp:242 -- BVN = cumnorm(-max(h, k)) with h = -a, k = -b.
            return min(_CND(a), _CND(b))
        if self._rho == -1.0:
            # cpp:244-255. After the rho < 0 sign flip the guard is
            # ``k > h`` i.e. ``a + b > 0``; below that the probability is
            # exactly zero. The two branches are algebraically the same
            # ``N(a) + N(b) - 1``; C++ picks whichever one evaluates
            # cumnorm in its accurate lower tail, and so does this.
            if a + b <= 0.0:
                return 0.0
            if a <= 0.0:
                return _CND(a) - _CND(-b)
            return _CND(b) - _CND(-a)

        # scipy stubs for ``multivariate_normal.cdf`` are incomplete in
        # current scipy-stubs: ``cov`` is declared as int rather than
        # array-like. Cast to ``Any`` and discard the unknown return type
        # via ``float()``.
        cdf_fn = cast(Any, _mvn).cdf
        return float(cdf_fn([a, b], mean=_MEAN_ZERO, cov=self._cov))


# C++ typedef ``BivariateCumulativeNormalDistribution`` is the default
# (= We04DP); the Dr78 6-decimal-place variant is the lower-precision
# legacy variant. We expose the same single implementation under both
# names since scipy's algorithm is at least as accurate as either.
BivariateCumulativeNormalDistributionDr78 = BivariateCumulativeNormalDistribution


__all__ = [
    "BivariateCumulativeNormalDistribution",
    "BivariateCumulativeNormalDistributionDr78",
]
