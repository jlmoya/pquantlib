"""LfmHullWhiteParameterization — LFM covariance from a caplet volatility curve.

# C++ parity: ql/legacy/libormarketmodels/lfmhullwhiteparam.{hpp,cpp} (v1.43).

Reference: Hull, John, White, Alan, 1999, *Forward Rate Volatilities, Swap
Rate Volatilities and the Implementation of the Libor Market Model*.

The parameterization bootstraps a piecewise-constant, time-homogeneous
volatility ladder ``lambda`` out of a caplet volatility term structure:

    var_i    = capletVol(T_i)^2 * yearFraction(T_0, T_i)
    cumVar_i = sum_{j=1}^{i-1} lambda_{i-j-1}^2 (t_{j+1} - t_j)
    lambda_{i-1} = sqrt((var_i - cumVar_i) / (t_1 - t_0))

and spreads it across ``factors`` Brownian drivers via the row-normalized
spectral pseudo-root of an optional correlation matrix.

Handle indirection: C++ takes ``ext::shared_ptr<OptionletVolatilityStructure>``
directly (not a Handle) — the caplet curve is threaded as a plain object here
too, so there is no divergence for this class.
"""

from __future__ import annotations

import bisect
import math

import numpy as np

from pquantlib import qassert
from pquantlib.legacy.libormarketmodels.lfm_covar_param import (
    LfmCovarianceParameterization,
)
from pquantlib.legacy.libormarketmodels.lfm_process import LiborForwardModelProcess
from pquantlib.legacy.libormarketmodels.lm_corr_model import spectral_pseudo_sqrt
from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix
from pquantlib.termstructures.volatility.optionlet.optionlet_volatility_structure import (
    OptionletVolatilityStructure,
)


class LfmHullWhiteParameterization(LfmCovarianceParameterization):
    """Hull-White LFM covariance parameterization.

    # C++ parity: ``class LfmHullWhiteParameterization``
    # (lfmhullwhiteparam.hpp:41-57).
    """

    def __init__(
        self,
        process: LiborForwardModelProcess,
        caplet_vol: OptionletVolatilityStructure,
        correlation: Matrix | None = None,
        factors: int = 1,
    ) -> None:
        """Bootstrap the volatility ladder.

        # C++ parity: lfmhullwhiteparam.cpp:25-87. C++ signals "no correlation
        # matrix" with an EMPTY ``Matrix()``; the Python default is ``None``.
        """
        super().__init__(process.size(), factors)
        self._fixing_times: list[float] = process.fixing_times()

        sqrt_corr = np.ones((self._size - 1, self._factors), dtype=np.float64)
        if correlation is None or np.asarray(correlation).size == 0:
            qassert.require(
                self._factors == 1,
                "correlation matrix must be given for multi factor models",
            )
        else:
            corr = np.asarray(correlation, dtype=np.float64)
            qassert.require(
                corr.shape[0] == self._size - 1 and corr.shape[0] == corr.shape[1],
                "wrong dimesion of the correlation matrix",
            )
            qassert.require(
                self._factors <= self._size - 1,
                "too many factors for given LFM process",
            )

            tmp_sqrt_corr = spectral_pseudo_sqrt(corr)

            # reduce to n factor model
            # "Reconstructing a valid correlation matrix from invalid data"
            # (<http://www.quarchome.org/correlationmatrix.pdf>)
            for i in range(self._size - 1):
                row = tmp_sqrt_corr[i, : self._factors]
                p = math.sqrt(float(np.dot(row, row)))
                sqrt_corr[i, :] = row / p

        self._diffusion: Matrix = np.zeros((self._size - 1, self._factors), dtype=np.float64)

        lambda_: list[float] = []
        fixing_times = process.fixing_times()
        fixing_dates = process.fixing_dates()

        for i in range(1, self._size):
            cum_var = 0.0
            for j in range(1, i):
                cum_var += (
                    lambda_[i - j - 1]
                    * lambda_[i - j - 1]
                    * (fixing_times[j + 1] - fixing_times[j])
                )

            vol = caplet_vol.volatility(fixing_dates[i], 0.0)
            var = vol * vol * caplet_vol.day_counter().year_fraction(
                fixing_dates[0], fixing_dates[i]
            )

            lambda_.append(
                math.sqrt((var - cum_var) / (fixing_times[1] - fixing_times[0]))
            )

            for q in range(self._factors):
                self._diffusion[i - 1, q] = sqrt_corr[i - 1, q] * lambda_[-1]

        self._covariance: Matrix = self._diffusion @ self._diffusion.T

    # --- protected --------------------------------------------------------

    def next_index_reset(self, t: float) -> int:
        """# C++ parity: lfmhullwhiteparam.cpp:90-93 — strict ``upper_bound``."""
        return bisect.bisect_right(self._fixing_times, t)

    # --- parameterization -------------------------------------------------

    def diffusion(self, t: float, x: Array | None = None) -> Matrix:
        """# C++ parity: lfmhullwhiteparam.cpp:96-106."""
        tmp = np.zeros((self._size, self._factors), dtype=np.float64)
        m = self.next_index_reset(t)
        for k in range(m, self._size):
            for q in range(self._factors):
                tmp[k, q] = self._diffusion[k - m, q]
        return tmp

    def covariance(self, t: float, x: Array | None = None) -> Matrix:
        """# C++ parity: lfmhullwhiteparam.cpp:108-119."""
        tmp = np.zeros((self._size, self._size), dtype=np.float64)
        m = self.next_index_reset(t)
        for k in range(m, self._size):
            for i in range(m, self._size):
                tmp[k, i] = self._covariance[k - m, i - m]
        return tmp

    def integrated_covariance(self, t: float, x: Array | None = None) -> Matrix:
        """Exact (piecewise-constant) integral of the covariance over [0, t].

        # C++ parity: lfmhullwhiteparam.cpp:121-141. Overrides the base
        # class's 64-segment numerical routine; the C++ test-suite
        # (testLambdaBootstrapping) checks the two agree to 1e-10.
        """
        tmp = np.zeros((self._size, self._size), dtype=np.float64)

        last = bisect.bisect_left(self._fixing_times, t)

        for i in range(last):
            dt = (self._fixing_times[i + 1] if i + 1 < last else t) - self._fixing_times[i]
            for k in range(i, self._size - 1):
                for level in range(i, self._size - 1):
                    tmp[k + 1, level + 1] += self._covariance[k - i, level - i] * dt

        return tmp


__all__ = ["LfmHullWhiteParameterization"]
