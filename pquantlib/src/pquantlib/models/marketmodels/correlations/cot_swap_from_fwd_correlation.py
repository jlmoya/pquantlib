"""CotSwapFromFwdCorrelation — coterminal-swap correlation from forward correlation.

# C++ parity:
# ql/models/marketmodels/correlations/cotswapfromfwdcorrelation.{hpp,cpp}
# (v1.42.1).

Derives the coterminal-swap-rate instantaneous correlation from a forward-rate
correlation structure. For each per-step forward correlation matrix ``C``, the
swap correlation is the correlation matrix extracted from the sandwich
``Z C Z^T``, where ``Z`` is the coterminal-swap-from-forward Z-matrix
(``SwapForwardMappings.coterminal_swap_zed_matrix``). Correlation coefficients
of expired rates (whose rate time has passed the step's correlation time) are
zeroed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix
from pquantlib.models.marketmodels.piecewise_constant_correlation import (
    PiecewiseConstantCorrelation,
)
from pquantlib.models.marketmodels.swap_forward_mappings import SwapForwardMappings

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.curve_state import CurveState


def _correlation_matrix_from_covariance(covariance: Matrix) -> Matrix:
    """Extract the correlation matrix from a (symmetric) covariance matrix.

    # C++ parity: ql/math/matrixutilities/getcovariance.hpp
    # CovarianceDecomposition::correlationMatrix (only that one accessor is
    # needed by CotSwapFromFwdCorrelation). Mirrors the C++ algorithm:
    # variances on the diagonal, then ``corr[i][j] = cov[i][j] /
    # sqrt(var_i * var_j)`` (lower-symmetric input; the upper half is
    # reflected).
    """
    cov = np.asarray(covariance, dtype=np.float64)
    size = cov.shape[0]
    variances = [cov[i, i] for i in range(size)]
    std_devs = [math.sqrt(v) for v in variances]
    correlation: Matrix = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        correlation[i, i] = 1.0
        for j in range(i):
            # only the lower symmetric part is used; reflect to upper
            cov_ij = 0.5 * (cov[i, j] + cov[j, i])
            denom = std_devs[i] * std_devs[j]
            value = cov_ij / denom if denom != 0.0 else 0.0
            correlation[i, j] = value
            correlation[j, i] = value
    return correlation


class CotSwapFromFwdCorrelation(PiecewiseConstantCorrelation):
    """Coterminal-swap correlation derived from a forward correlation.

    # C++ parity: cotswapfromfwdcorrelation.hpp/.cpp CotSwapFromFwdCorrelation.
    """

    def __init__(
        self,
        fwd_corr: PiecewiseConstantCorrelation,
        curve_state: CurveState,
        displacement: float,
    ) -> None:
        self._fwd_corr = fwd_corr
        self._number_of_rates = fwd_corr.number_of_rates()

        qassert.require(
            self._number_of_rates == curve_state.number_of_rates(),
            f"mismatch between number of rates in fwdCorr ({self._number_of_rates}) "
            f"and curveState ({curve_state.number_of_rates()})",
        )

        zed = SwapForwardMappings.coterminal_swap_zed_matrix(curve_state, displacement)
        zed_t = zed.T
        fwd_corr_matrices = fwd_corr.correlations()
        self._swap_corr_matrices: list[Matrix] = [
            np.zeros((self._number_of_rates, self._number_of_rates), dtype=np.float64)
            for _ in range(len(fwd_corr_matrices))
        ]
        rate_times = curve_state.rate_times()
        corr_times = fwd_corr.times()
        for k in range(len(fwd_corr_matrices)):
            sandwiched = zed @ fwd_corr_matrices[k] @ zed_t
            self._swap_corr_matrices[k] = _correlation_matrix_from_covariance(sandwiched)
            # zero expired rates' correlation coefficients
            for i in range(self._number_of_rates):
                for j in range(i + 1):
                    if corr_times[k] > rate_times[j]:
                        self._swap_corr_matrices[k][i, j] = 0.0
                        self._swap_corr_matrices[k][j, i] = 0.0

    def times(self) -> list[float]:
        """Interval boundary times (delegated to the forward correlation).

        # C++ parity: CotSwapFromFwdCorrelation::times.
        """
        return self._fwd_corr.times()

    def rate_times(self) -> list[float]:
        """The rate times (delegated to the forward correlation).

        # C++ parity: CotSwapFromFwdCorrelation::rateTimes.
        """
        return self._fwd_corr.rate_times()

    def correlations(self) -> list[Matrix]:
        """Per-interval swap correlation matrices.

        # C++ parity: CotSwapFromFwdCorrelation::correlations.
        """
        return self._swap_corr_matrices

    def number_of_rates(self) -> int:
        """Number of forward rates.

        # C++ parity: CotSwapFromFwdCorrelation::numberOfRates.
        """
        return self._number_of_rates
