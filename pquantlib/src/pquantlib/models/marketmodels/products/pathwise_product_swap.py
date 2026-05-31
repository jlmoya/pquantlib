"""MarketModelPathwiseSwap — pathwise (Greeks-aware) swap.

# C++ parity:
# ql/models/marketmodels/products/pathwise/pathwiseproductswap.{hpp,cpp}
# (v1.42.1).

A single product. At step ``i`` it pays ``(libor_i - strike_i) * accrual_i *
multiplier`` at payment-time index ``i+1`` (the rate-time grid is used as the
payment-time vector, with ``rateTimes[0]`` kept for index bookkeeping). The only
non-zero derivative is ``accrual_i * multiplier`` w.r.t. forward ``i``. Fairly
useless directly, but a building block for breakable swaps.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.pathwise_multi_product import (
    MarketModelPathwiseMultiProduct,
    PathwiseCashFlow,
)
from pquantlib.models.marketmodels.utilities import check_increasing_times


class MarketModelPathwiseSwap(MarketModelPathwiseMultiProduct):
    """A pathwise (Greeks-aware) vanilla swap.

    # C++ parity: pathwiseproductswap.hpp MarketModelPathwiseSwap.
    """

    def __init__(
        self,
        rate_times: list[float],
        accruals: list[float],
        strikes: list[float],
        multiplier: float = 1.0,
    ) -> None:
        self._rate_times = list(rate_times)
        self._accruals = list(accruals)
        self._strikes = list(strikes)
        self._number_rates = len(rate_times) - 1
        self._multiplier = multiplier
        self._current_index = 0
        check_increasing_times(rate_times)
        evol_times = list(self._rate_times[:-1])
        qassert.require(
            len(evol_times) == self._number_rates, "rateTimes.size()<> numberOfRates+1"
        )
        if len(self._strikes) == 1:
            self._strikes = [self._strikes[0]] * self._number_rates
        if len(self._accruals) == 1:
            self._accruals = [self._accruals[0]] * self._number_rates
        qassert.require(
            len(self._accruals) == self._number_rates,
            "accruals.size() does not equal numberOfRates or 1",
        )
        qassert.require(
            len(self._strikes) == self._number_rates,
            "strikes.size() does not equal numberOfRates or 1",
        )
        self._evolution = EvolutionDescription(rate_times, evol_times)

    def already_deflated(self) -> bool:
        return False

    def suggested_numeraires(self) -> list[int]:
        return list(range(self._number_rates))

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def possible_cash_flow_times(self) -> list[float]:
        # rateTimes_[0] is not used as a cash-flow time but kept for bookkeeping.
        return self._rate_times

    def number_of_products(self) -> int:
        return 1

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return 1

    def reset(self) -> None:
        self._current_index = 0

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[PathwiseCashFlow]],
    ) -> bool:
        i = self._current_index
        libor_rate = current_state.forward_rate(i)
        cf = cash_flows_generated[0][0]
        cf.time_index = i + 1
        cf.amount[0] = (libor_rate - self._strikes[i]) * self._accruals[i] * self._multiplier

        number_cash_flows_this_step[0] = 1

        for k in range(1, self._number_rates + 1):
            cf.amount[k] = 0.0
        cf.amount[i + 1] = self._accruals[i] * self._multiplier

        self._current_index += 1
        return self._current_index == len(self._strikes)

    def clone(self) -> MarketModelPathwiseMultiProduct:
        return MarketModelPathwiseSwap(
            self._rate_times, self._accruals, self._strikes, self._multiplier
        )
