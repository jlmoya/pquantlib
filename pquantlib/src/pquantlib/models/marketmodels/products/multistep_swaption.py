"""MultiStepSwaption — a single swaption on a contiguous subset of rates.

# C++ parity:
# ql/models/marketmodels/products/multistep/multistepswaption.{hpp,cpp}
# (v1.42.1).

Prices a swaption on the constant-maturity swap spanning rate indices
``[startIndex, endIndex)``. Useful only for testing. The product steps through
all rate times up to the swap start; at ``startIndex`` it pays
``payoff(cmSwapRate) * cmSwapAnnuity`` at the single payment-time index 0.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.multi_product import (
    CashFlow,
    MarketModelMultiProduct,
)
from pquantlib.models.marketmodels.products.multiproduct_multistep import (
    MultiProductMultiStep,
)
from pquantlib.payoffs import StrikedTypePayoff


class MultiStepSwaption(MultiProductMultiStep):
    """A single swaption on a contiguous subset of rates.

    # C++ parity: multistepswaption.hpp MultiStepSwaption.
    """

    def __init__(
        self,
        rate_times: list[float],
        start_index: int,
        end_index: int,
        payoff: StrikedTypePayoff,
    ) -> None:
        super().__init__(rate_times)
        qassert.require(
            start_index < end_index, " start index must be before end index"
        )
        qassert.require(
            end_index < len(rate_times),
            "end index be before the end of the rates.",
        )
        self._start_index = start_index
        self._end_index = end_index
        self._payoff = payoff
        self._payment_times = [rate_times[start_index]]
        self._current_index = 0

    def possible_cash_flow_times(self) -> list[float]:
        return self._payment_times

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
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        if self._current_index == self._start_index:
            cf = cash_flows_generated[0][0]
            cf.time_index = 0
            swap_rate = current_state.cm_swap_rate(
                self._start_index, self._end_index - self._start_index
            )
            annuity = current_state.cm_swap_annuity(
                self._start_index,
                self._start_index,
                self._end_index - self._start_index,
            )
            cf.amount = self._payoff(swap_rate) * annuity
            number_cash_flows_this_step[0] = 1 if cf.amount != 0.0 else 0
            return True
        number_cash_flows_this_step[0] = 0
        self._current_index += 1
        return False

    def clone(self) -> MarketModelMultiProduct:
        return MultiStepSwaption(
            self._rate_times, self._start_index, self._end_index, self._payoff
        )
