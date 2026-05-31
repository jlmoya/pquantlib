"""TimeHomogeneousForwardCorrelation — time-homogeneous evolved correlation.

# C++ parity:
# ql/models/marketmodels/correlations/timehomogeneousforwardcorrelation.{hpp,cpp}
# (v1.42.1).

Wraps a single forward-rate correlation matrix into the time-homogeneous
family of per-step instantaneous-correlation matrices. At step ``k`` the
matrix is the original forward correlation shifted into its lower-right
``(n - k)`` block (so as rates expire, the structure slides), with unit
diagonal on the alive rates. ``evolved_matrices`` is the static builder; the
class wraps it as a ``PiecewiseConstantCorrelation``.
"""

from __future__ import annotations

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix
from pquantlib.models.marketmodels.piecewise_constant_correlation import (
    PiecewiseConstantCorrelation,
)
from pquantlib.models.marketmodels.utilities import check_increasing_times


class TimeHomogeneousForwardCorrelation(PiecewiseConstantCorrelation):
    """Time-homogeneous forward-correlation structure.

    # C++ parity: timehomogeneousforwardcorrelation.hpp/.cpp
    TimeHomogeneousForwardCorrelation.
    """

    def __init__(self, fwd_correlation: Matrix, rate_times: list[float]) -> None:
        self._number_of_rates = 0 if not rate_times else len(rate_times) - 1
        self._fwd_correlation: Matrix = np.asarray(fwd_correlation, dtype=np.float64)
        self._rate_times = list(rate_times)

        check_increasing_times(rate_times)
        qassert.require(
            self._number_of_rates >= 1,
            "Rate times must contain at least two values",
        )
        qassert.require(
            self._number_of_rates == self._fwd_correlation.shape[0],
            f"mismatch between number of rates ({self._number_of_rates}) and "
            f"fwdCorrelation rows ({self._fwd_correlation.shape[0]})",
        )
        qassert.require(
            self._number_of_rates == self._fwd_correlation.shape[1],
            f"mismatch between number of rates ({self._number_of_rates}) and "
            f"fwdCorrelation columns ({self._fwd_correlation.shape[1]})",
        )

        # times_ = rateTimes[:-1]
        self._times = list(rate_times[:-1])
        self._correlations = self.evolved_matrices(self._fwd_correlation)

    @staticmethod
    def evolved_matrices(fwd_correlation: Matrix) -> list[Matrix]:
        """Time-homogeneous family of per-step correlation matrices.

        # C++ parity: timehomogeneousforwardcorrelation.cpp
        TimeHomogeneousForwardCorrelation::evolvedMatrices.
        """
        fwd = np.asarray(fwd_correlation, dtype=np.float64)
        number_of_rates = fwd.shape[0]
        correlations: list[Matrix] = [
            np.zeros((number_of_rates, number_of_rates), dtype=np.float64)
            for _ in range(number_of_rates)
        ]
        for k in range(number_of_rates):
            # proper diagonal values
            for i in range(k, number_of_rates):
                correlations[k][i, i] = 1.0
            # copy only time homogeneous values
            for i in range(k, number_of_rates):
                for j in range(k, i):
                    value = fwd[i - k, j - k]
                    correlations[k][i, j] = value
                    correlations[k][j, i] = value
        return correlations

    def times(self) -> list[float]:
        """Interval boundary times.

        # C++ parity: TimeHomogeneousForwardCorrelation::times.
        """
        return self._times

    def rate_times(self) -> list[float]:
        """The rate times.

        # C++ parity: TimeHomogeneousForwardCorrelation::rateTimes.
        """
        return self._rate_times

    def correlations(self) -> list[Matrix]:
        """Per-interval correlation matrices.

        # C++ parity: TimeHomogeneousForwardCorrelation::correlations.
        """
        return self._correlations

    def number_of_rates(self) -> int:
        """Number of forward rates.

        # C++ parity: TimeHomogeneousForwardCorrelation::numberOfRates.
        """
        return self._number_of_rates
