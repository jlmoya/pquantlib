"""MarketModelPathwiseDiscounter — pathwise (Greeks-aware) discounter.

# C++ parity: ql/models/marketmodels/pathwisediscounter.{hpp,cpp} (v1.42.1).

Returns the number of units of the discretely-compounding money-market
account that one unit of cash at the payment time can buy, *plus* the
derivative of that number with respect to each forward rate. Discounting is
purely on the simulation LIBOR rates; to discount back to time zero multiply
by ``P(t_0)``.
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

from pquantlib.models.marketmodels.utilities import check_increasing_times

if TYPE_CHECKING:
    from pquantlib.math.matrix import Matrix


class MarketModelPathwiseDiscounter:
    """Pathwise discounter: discount factor + per-forward derivatives.

    # C++ parity: pathwisediscounter.hpp MarketModelPathwiseDiscounter.
    """

    def __init__(self, payment_time: float, rate_times: list[float]) -> None:
        check_increasing_times(rate_times)
        self._number_rates = len(rate_times) - 1
        self._before = bisect.bisect_left(rate_times, payment_time)
        self._before = min(self._before, len(rate_times) - 2)
        self._before_weight = 1.0 - (payment_time - rate_times[self._before]) / (
            rate_times[self._before + 1] - rate_times[self._before]
        )
        self._post_weight = 1.0 - self._before_weight
        self._taus = [rate_times[i + 1] - rate_times[i] for i in range(self._number_rates)]

    def get_factors(
        self,
        libor_rates: Matrix,
        discounts: Matrix,
        current_step: int,
        factors: list[float],
    ) -> None:
        """Fill ``factors`` with the discount + its per-forward derivatives.

        # C++ parity: pathwisediscounter.cpp MarketModelPathwiseDiscounter::getFactors.

        ``discounts[current_step][j]`` is ``P(t_0, t_j)`` for the step;
        ``factors`` (length ``number_rates + 1``) is a caller-supplied output
        buffer: ``factors[0]`` is the discount, ``factors[i+1]`` its
        derivative w.r.t. forward rate ``i``. ``libor_rates`` is accepted for
        signature parity but unused (as in the C++ source).
        """
        pre_df = float(discounts[current_step, self._before])
        post_df = float(discounts[current_step, self._before + 1])

        for i in range(self._before + 1, self._number_rates):
            factors[i + 1] = 0.0

        if self._post_weight == 0.0:
            factors[0] = pre_df
            for i in range(self._before):
                factors[i + 1] = (
                    -pre_df
                    * self._taus[i]
                    * float(discounts[current_step, i + 1])
                    / float(discounts[current_step, i])
                )
            factors[self._before + 1] = 0.0
            return

        df = pre_df * (post_df / pre_df) ** self._post_weight
        factors[0] = df
        for i in range(self._before + 1):
            factors[i + 1] = (
                -df
                * self._taus[i]
                * float(discounts[current_step, i + 1])
                / float(discounts[current_step, i])
            )
        factors[self._before + 1] *= self._post_weight
