"""MultiStepCoterminalSwaptions — coterminal swaptions as a product.

# C++ parity:
# ql/models/marketmodels/products/multistep/multistepcoterminalswaptions.{hpp,cpp}
# (v1.42.1).

``lastIndex`` products, one per coterminal swaption. At step ``i`` product ``i``
pays ``payoff[i](coterminalSwapRate(i)) * coterminalSwapAnnuity(i, i)`` at
payment-time index ``i``.
"""

from __future__ import annotations

from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.multi_product import (
    CashFlow,
    MarketModelMultiProduct,
)
from pquantlib.models.marketmodels.products.multiproduct_multistep import (
    MultiProductMultiStep,
)
from pquantlib.models.marketmodels.utilities import check_increasing_times
from pquantlib.payoffs import StrikedTypePayoff


class MultiStepCoterminalSwaptions(MultiProductMultiStep):
    """The family of coterminal swaptions.

    # C++ parity: multistepcoterminalswaptions.hpp MultiStepCoterminalSwaptions.
    """

    def __init__(
        self,
        rate_times: list[float],
        payment_times: list[float],
        payoffs: list[StrikedTypePayoff],
    ) -> None:
        super().__init__(rate_times)
        self._payment_times = list(payment_times)
        self._payoffs = list(payoffs)
        self._last_index = len(rate_times) - 1
        self._current_index = 0
        check_increasing_times(payment_times)

    def possible_cash_flow_times(self) -> list[float]:
        return self._payment_times

    def number_of_products(self) -> int:
        return self._last_index

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
        cf = cash_flows_generated[self._current_index][0]
        cf.time_index = self._current_index

        swap_rate = current_state.coterminal_swap_rate(self._current_index)
        annuity = current_state.coterminal_swap_annuity(
            self._current_index, self._current_index
        )
        cf.amount = self._payoffs[self._current_index](swap_rate) * annuity
        for i in range(len(number_cash_flows_this_step)):
            number_cash_flows_this_step[i] = 0
        number_cash_flows_this_step[self._current_index] = 1
        self._current_index += 1
        return self._current_index == self._last_index

    def clone(self) -> MarketModelMultiProduct:
        return MultiStepCoterminalSwaptions(
            self._rate_times, self._payment_times, self._payoffs
        )
