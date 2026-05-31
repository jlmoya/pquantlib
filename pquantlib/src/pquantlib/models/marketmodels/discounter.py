"""MarketModelDiscounter — numeraire-rebased log-linear discounter.

# C++ parity: ql/models/marketmodels/discounter.{hpp,cpp} (v1.42.1).

Given a payment time and the rate-time grid, this returns the number of
units of the chosen numeraire bond that one unit of cash at the payment
time can buy, log-linearly interpolating between the two bracketing
discount bonds of the supplied ``CurveState``.
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

from pquantlib.models.marketmodels.utilities import check_increasing_times

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.curve_state import CurveState


class MarketModelDiscounter:
    """Numeraire-rebased discounter for a fixed payment time.

    # C++ parity: discounter.hpp MarketModelDiscounter.
    """

    def __init__(self, payment_time: float, rate_times: list[float]) -> None:
        check_increasing_times(rate_times)
        # lower_bound: first index with rate_times[idx] >= payment_time
        self._before = bisect.bisect_left(rate_times, payment_time)
        # handle payment in/after the last period
        self._before = min(self._before, len(rate_times) - 2)
        self._before_weight = 1.0 - (payment_time - rate_times[self._before]) / (
            rate_times[self._before + 1] - rate_times[self._before]
        )

    def numeraire_bonds(self, curve_state: CurveState, numeraire: int) -> float:
        """Numeraire bonds bought by one unit of cash at the payment time.

        # C++ parity: discounter.cpp MarketModelDiscounter::numeraireBonds.
        """
        pre_df = curve_state.discount_ratio(self._before, numeraire)
        if self._before_weight == 1.0:
            return pre_df
        post_df = curve_state.discount_ratio(self._before + 1, numeraire)
        if self._before_weight == 0.0:
            return post_df
        return pre_df**self._before_weight * post_df ** (1.0 - self._before_weight)
