"""LfmCovarianceParameterization — abstract LFM covariance parameterization.

# C++ parity: ql/legacy/libormarketmodels/lfmcovarparam.{hpp,cpp} (v1.43).

Reference: Brigo, Mercurio, Morini, 2003, *Different Covariance
Parameterizations of the Libor Market Model and Joint Caps/Swaptions
Calibration*.

Subclasses supply ``diffusion(t, x)``; the base derives the instantaneous
covariance from it and provides a brute-force integrated covariance (64
adaptive Gauss-Kronrod segments per matrix entry). The C++ comment on that
routine is explicit that it is "not intended for production" and exists for
testing and R&D — it is nevertheless the cross-check the C++ test-suite runs
:class:`LfmHullWhiteParameterization` against, so it is ported faithfully.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.integrals.kronrod import GaussKronrodAdaptive
from pquantlib.math.matrix import Matrix


class LfmCovarianceParameterization(ABC):
    """Abstract covariance parameterization of a LIBOR forward model.

    # C++ parity: ``class LfmCovarianceParameterization``
    # (lfmcovarparam.hpp:39-58).
    """

    def __init__(self, size: int, factors: int) -> None:
        # C++ parity: lfmcovarparam.hpp:41-42.
        self._size: int = size
        self._factors: int = factors

    # --- inspectors -------------------------------------------------------

    def size(self) -> int:
        """# C++ parity: lfmcovarparam.hpp:45."""
        return self._size

    def factors(self) -> int:
        """# C++ parity: lfmcovarparam.hpp:46."""
        return self._factors

    # --- parameterization -------------------------------------------------

    @abstractmethod
    def diffusion(self, t: float, x: Array | None = None) -> Matrix:
        """The size x factors diffusion matrix at time ``t``.

        # C++ parity: pure virtual (lfmcovarparam.hpp:48).
        """

    def covariance(self, t: float, x: Array | None = None) -> Matrix:
        """Instantaneous covariance ``sigma sigma^T``.

        # C++ parity: ``LfmCovarianceParameterization::covariance``
        # (lfmcovarparam.cpp:47-51).
        """
        sigma = self.diffusion(t, x)
        result: Matrix = sigma @ sigma.T
        return result

    def integrated_covariance(self, t: float, x: Array | None = None) -> Matrix:
        """Integral of the instantaneous covariance over [0, t], numerically.

        # C++ parity: ``LfmCovarianceParameterization::integratedCovariance``
        # (lfmcovarparam.cpp:53-74) — for each lower-triangular (i, j) it sums
        # ``GaussKronrodAdaptive(1e-10, 10000)`` over 64 equal sub-intervals of
        # [0, t], where the integrand is the inner product of rows i and j of
        # ``diffusion(u)``. The upper triangle is mirrored.

        One integrator is shared across the 64 sub-intervals, as in C++; both
        ports reset the evaluation counter at the top of every ``__call__`` /
        ``operator()``, so the 10000-evaluation budget is per sub-interval.
        """
        qassert.require(x is None or np.asarray(x).size == 0, "can not handle given x here")

        tmp = np.zeros((self._size, self._size), dtype=np.float64)

        for i in range(self._size):
            for j in range(i + 1):

                def helper(u: float, i: int = i, j: int = j) -> float:
                    # C++ parity: LfmCovarianceParameterization::Var_Helper
                    # (lfmcovarparam.cpp:40-45).
                    m = self.diffusion(u)
                    return float(np.dot(m[i], m[j]))

                integrator = GaussKronrodAdaptive(1e-10, 10000)
                total = 0.0
                for k in range(64):
                    total += integrator(helper, k * t / 64.0, (k + 1) * t / 64.0)
                tmp[i, j] = total
                tmp[j, i] = total

        return tmp


__all__ = ["LfmCovarianceParameterization"]
