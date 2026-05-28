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

_MEAN_ZERO: Final[list[float]] = [0.0, 0.0]


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
