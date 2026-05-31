"""MultiStepCoterminalSwaps — the family of coterminal swaps as a product.

# C++ parity:
# ql/models/marketmodels/products/multistep/multistepcoterminalswaps.{hpp,cpp}
# (v1.42.1).

``lastIndex`` products, one per coterminal swap. At step ``i`` every product
``j <= i`` accrues the step-``i`` fixed/floating cash flows (so the longest swap,
product 0, receives all of them; the shortest only the last). All paid at
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


class MultiStepCoterminalSwaps(MultiProductMultiStep):
    """The family of coterminal swaps.

    # C++ parity: multistepcoterminalswaps.hpp MultiStepCoterminalSwaps.
    """

    def __init__(
        self,
        rate_times: list[float],
        fixed_accruals: list[float],
        floating_accruals: list[float],
        payment_times: list[float],
        fixed_rate: float,
    ) -> None:
        super().__init__(rate_times)
        self._fixed_accruals = list(fixed_accruals)
        self._floating_accruals = list(floating_accruals)
        self._payment_times = list(payment_times)
        self._fixed_rate = fixed_rate
        self._last_index = len(rate_times) - 1
        self._current_index = 0
        check_increasing_times(payment_times)

    def possible_cash_flow_times(self) -> list[float]:
        return self._payment_times

    def number_of_products(self) -> int:
        return self._last_index

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return 2

    def reset(self) -> None:
        self._current_index = 0

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        libor_rate = current_state.forward_rate(self._current_index)
        for i in range(len(number_cash_flows_this_step)):
            number_cash_flows_this_step[i] = 0
        for i in range(self._current_index + 1):
            fixed = cash_flows_generated[i][0]
            fixed.time_index = self._current_index
            fixed.amount = -self._fixed_rate * self._fixed_accruals[self._current_index]

            floating = cash_flows_generated[i][1]
            floating.time_index = self._current_index
            floating.amount = libor_rate * self._floating_accruals[self._current_index]

            number_cash_flows_this_step[i] = 2
        self._current_index += 1
        return self._current_index == self._last_index

    def clone(self) -> MarketModelMultiProduct:
        return MultiStepCoterminalSwaps(
            self._rate_times,
            self._fixed_accruals,
            self._floating_accruals,
            self._payment_times,
            self._fixed_rate,
        )
