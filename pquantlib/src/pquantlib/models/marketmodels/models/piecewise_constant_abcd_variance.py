"""PiecewiseConstantAbcdVariance — abcd-form piecewise-constant variance.

# C++ parity:
# ql/models/marketmodels/models/piecewiseconstantabcdvariance.{hpp,cpp}
# (v1.42.1).

The variance of a single (``resetIndex``) forward rate, decomposed
interval-by-interval over the rate-time grid, using the Rebonato abcd
instantaneous-volatility form. For interval ``i`` (``i <= resetIndex``) the
variance is the integral of the squared abcd vol of the ``resetIndex``-fixing
rate over ``[rateTimes[i-1], rateTimes[i]]``.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.models.marketmodels.models.abcd_function import AbcdFunction
from pquantlib.models.marketmodels.models.piecewise_constant_variance import (
    PiecewiseConstantVariance,
)
from pquantlib.models.marketmodels.utilities import check_increasing_times


class PiecewiseConstantAbcdVariance(PiecewiseConstantVariance):
    """Abcd-form piecewise-constant variance of a single forward rate.

    # C++ parity: piecewiseconstantabcdvariance.hpp/.cpp
    PiecewiseConstantAbcdVariance.
    """

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        d: float,
        reset_index: int,
        rate_times: list[float],
    ) -> None:
        n = len(rate_times) - 1
        self._variances: list[float] = [0.0] * n
        self._volatilities: list[float] = [0.0] * n
        self._rate_times = list(rate_times)
        self._a = a
        self._b = b
        self._c = c
        self._d = d

        check_increasing_times(rate_times)
        qassert.require(
            len(rate_times) > 1,
            "Rate times must contain at least two values",
        )
        qassert.require(
            reset_index < len(self._rate_times) - 1,
            f"resetIndex ({reset_index}) must be less than rateTimes.size()-1 "
            f"({len(self._rate_times) - 1})",
        )
        abcd = AbcdFunction(a, b, c, d)
        for i in range(reset_index + 1):
            start_time = 0.0 if i == 0 else self._rate_times[i - 1]
            self._variances[i] = abcd.variance(
                start_time, self._rate_times[i], self._rate_times[reset_index]
            )
            tot_time = self._rate_times[i] - start_time
            self._volatilities[i] = math.sqrt(self._variances[i] / tot_time)

    def get_abcd(self) -> tuple[float, float, float, float]:
        """Return the ``(a, b, c, d)`` parameters.

        # C++ parity: PiecewiseConstantAbcdVariance::getABCD (out-params in C++;
        a returned tuple here).
        """
        return (self._a, self._b, self._c, self._d)

    def rate_times(self) -> list[float]:
        """The rate times.

        # C++ parity: PiecewiseConstantAbcdVariance::rateTimes.
        """
        return self._rate_times

    def variances(self) -> list[float]:
        """The per-step variances.

        # C++ parity: PiecewiseConstantAbcdVariance::variances.
        """
        return self._variances

    def volatilities(self) -> list[float]:
        """The per-step volatilities.

        # C++ parity: PiecewiseConstantAbcdVariance::volatilities.
        """
        return self._volatilities
