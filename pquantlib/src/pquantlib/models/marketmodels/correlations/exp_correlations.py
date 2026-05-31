"""Exponential instantaneous correlation + ExponentialForwardCorrelation.

# C++ parity: ql/models/marketmodels/correlations/expcorrelations.{hpp,cpp}
# (v1.42.1).

``exponential_forward_correlation`` is the C++ free function
``exponentialCorrelations`` — the closed-form exponential instantaneous
correlation between (alive) forward rates:

    corr(i, j) = L + (1 - L) * exp(-beta * |(t_i - t)^gamma - (t_j - t)^gamma|)

where ``L`` is the long-term correlation, ``beta`` the exponential decay,
``gamma`` the time-to-go exponent, and ``t`` the evaluation time (only rates
with ``t <= t_k`` are alive; expired rates get zero correlation).

``ExponentialForwardCorrelation`` is the ``PiecewiseConstantCorrelation``
built from it. With ``gamma == 1`` the structure is time-homogeneous and the
per-step matrices are produced by
``TimeHomogeneousForwardCorrelation.evolved_matrices``; otherwise each step's
matrix is the closed-form correlation evaluated at the interval midpoint.

# C++ parity: the C++ function is named ``exponentialCorrelations``; the
# brief's required public name is ``exponential_forward_correlation``. We
# expose that name and keep ``exponential_correlations`` as an alias so both
# the C++ spelling and the brief spelling resolve.
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib import qassert
from pquantlib.math.closeness import close
from pquantlib.math.matrix import Matrix
from pquantlib.models.marketmodels.correlations.time_homogeneous_forward_correlation import (
    TimeHomogeneousForwardCorrelation,
)
from pquantlib.models.marketmodels.piecewise_constant_correlation import (
    PiecewiseConstantCorrelation,
)
from pquantlib.models.marketmodels.utilities import check_increasing_times


def exponential_forward_correlation(
    rate_times: list[float],
    long_term_corr: float = 0.5,
    beta: float = 0.2,
    gamma: float = 1.0,
    times: float = 0.0,
) -> Matrix:
    """Closed-form exponential instantaneous correlation matrix.

    # C++ parity: expcorrelations.cpp exponentialCorrelations.

    ``times`` is the scalar evaluation time ``t`` (C++ parameter ``time``);
    the brief names the kwarg ``times`` to mirror the header signature.
    """
    # preliminary checks
    check_increasing_times(rate_times)
    qassert.require(
        long_term_corr <= 1.0 and long_term_corr >= 0.0,
        f"Long term correlation ({long_term_corr}) outside [0;1] interval",
    )
    qassert.require(beta >= 0.0, f"beta ({beta}) must be greater than zero")
    qassert.require(
        gamma <= 1.0 and gamma >= 0.0,
        f"gamma ({gamma}) outside [0;1] interval",
    )

    time = times
    nb_rows = len(rate_times) - 1
    correlations: Matrix = np.zeros((nb_rows, nb_rows), dtype=np.float64)
    for i in range(nb_rows):
        # correlation is defined only between (alive) stochastic rates...
        if time <= rate_times[i]:
            correlations[i, i] = 1.0
            for j in range(i):
                if time <= rate_times[j]:
                    value = long_term_corr + (1.0 - long_term_corr) * math.exp(
                        -beta
                        * abs(
                            math.pow(rate_times[i] - time, gamma)
                            - math.pow(rate_times[j] - time, gamma)
                        )
                    )
                    correlations[i, j] = value
                    correlations[j, i] = value
    return correlations


# C++ spelling alias.
exponential_correlations = exponential_forward_correlation


class ExponentialForwardCorrelation(PiecewiseConstantCorrelation):
    """Piecewise-constant exponential forward correlation structure.

    # C++ parity: expcorrelations.hpp/.cpp ExponentialForwardCorrelation.
    """

    def __init__(
        self,
        rate_times: list[float],
        long_term_corr: float = 0.5,
        beta: float = 0.2,
        gamma: float = 1.0,
        times: list[float] | None = None,
    ) -> None:
        self._number_of_rates = 0 if not rate_times else len(rate_times) - 1
        self._long_term_corr = long_term_corr
        self._beta = beta
        self._gamma = gamma
        self._rate_times = list(rate_times)
        self._times: list[float] = [] if times is None else list(times)
        self._correlations: list[Matrix] = []

        qassert.require(
            self._number_of_rates > 1,
            "Rate times must contain at least two values",
        )

        check_increasing_times(self._rate_times)

        # corrTimes must include all rateTimes but the last
        if not self._times:
            self._times = list(self._rate_times[:-1])
        else:
            check_increasing_times(self._times)

        if close(gamma, 1.0):
            temp = list(self._rate_times[:-1])
            qassert.require(
                self._times == temp,
                f"corr times {self._times} must be equal to (all) rate times "
                f"(but the last) {temp}",
            )
            c = exponential_forward_correlation(
                self._rate_times, self._long_term_corr, self._beta, 1.0, 0.0
            )
            self._correlations = TimeHomogeneousForwardCorrelation.evolved_matrices(c)
        else:
            # FIXME (C++): should check here that all rateTimes but the last
            # are included in rateTimes
            qassert.require(
                self._times[-1] <= self._rate_times[self._number_of_rates],
                f"last corr time {self._times[-1]} is after next-to-last rate "
                f"time {self._rate_times[self._number_of_rates]}",
            )
            self._correlations = []
            time = self._times[0] / 2.0
            self._correlations.append(
                exponential_forward_correlation(
                    self._rate_times, self._long_term_corr, self._beta, self._gamma, time
                )
            )
            for k in range(1, len(self._times)):
                time = (self._times[k] + self._times[k - 1]) / 2.0
                self._correlations.append(
                    exponential_forward_correlation(
                        self._rate_times,
                        self._long_term_corr,
                        self._beta,
                        self._gamma,
                        time,
                    )
                )

    def times(self) -> list[float]:
        """Interval boundary times.

        # C++ parity: ExponentialForwardCorrelation::times.
        """
        return self._times

    def rate_times(self) -> list[float]:
        """The rate times.

        # C++ parity: ExponentialForwardCorrelation::rateTimes.
        """
        return self._rate_times

    def correlations(self) -> list[Matrix]:
        """Per-interval correlation matrices.

        # C++ parity: ExponentialForwardCorrelation::correlations.
        """
        return self._correlations

    def number_of_rates(self) -> int:
        """Number of forward rates.

        # C++ parity: ExponentialForwardCorrelation::numberOfRates.
        """
        return self._number_of_rates
