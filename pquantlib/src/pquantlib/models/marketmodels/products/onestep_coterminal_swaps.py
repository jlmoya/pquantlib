"""OneStepCoterminalSwaps — the family of coterminal swaps in a single step.

# C++ parity:
# ql/models/marketmodels/products/onestep/onestepcoterminalswaps.{hpp,cpp}
# (v1.42.1).

``lastIndex`` products, one per coterminal swap. In the single step every
rate-time ``indexOfTime`` contributes a fixed/floating cash-flow pair to every
product ``i <= indexOfTime``, packed into product ``i``'s cash-flow slots
``2*(indexOfTime-i)`` / ``2*(indexOfTime-i)+1``.
"""

from __future__ import annotations

from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.multi_product import (
    CashFlow,
    MarketModelMultiProduct,
)
from pquantlib.models.marketmodels.products.multiproduct_onestep import (
    MultiProductOneStep,
)
from pquantlib.models.marketmodels.utilities import check_increasing_times


class OneStepCoterminalSwaps(MultiProductOneStep):
    """The family of coterminal swaps evaluated in a single step.

    # C++ parity: onestepcoterminalswaps.hpp OneStepCoterminalSwaps.
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
        check_increasing_times(payment_times)

    def possible_cash_flow_times(self) -> list[float]:
        return self._payment_times

    def number_of_products(self) -> int:
        return self._last_index

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return 2 * self._last_index

    def reset(self) -> None:
        # nothing to do
        pass

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        for k in range(len(number_cash_flows_this_step)):
            number_cash_flows_this_step[k] = 0

        for index_of_time in range(self._last_index):
            libor_rate = current_state.forward_rate(index_of_time)
            for i in range(index_of_time + 1):
                fixed = cash_flows_generated[i][(index_of_time - i) * 2]
                fixed.time_index = index_of_time
                fixed.amount = -self._fixed_rate * self._fixed_accruals[index_of_time]

                floating = cash_flows_generated[i][(index_of_time - i) * 2 + 1]
                floating.time_index = index_of_time
                floating.amount = libor_rate * self._floating_accruals[index_of_time]

                number_cash_flows_this_step[i] += 2
        return True

    def clone(self) -> MarketModelMultiProduct:
        return OneStepCoterminalSwaps(
            self._rate_times,
            self._fixed_accruals,
            self._floating_accruals,
            self._payment_times,
            self._fixed_rate,
        )
