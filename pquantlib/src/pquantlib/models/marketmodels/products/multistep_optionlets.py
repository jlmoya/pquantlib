"""MultiStepOptionlets — caplets / floorlets as a multi-step product.

# C++ parity: ql/models/marketmodels/products/multistep/multistepoptionlets.{hpp,cpp}
# (v1.42.1).

The canonical BGM test product. One product per payoff. At step ``i`` product
``i`` pays ``payoff[i](forwardRate(i)) * accrual[i]`` at payment-time index
``i``.
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
from pquantlib.payoffs import Payoff


class MultiStepOptionlets(MultiProductMultiStep):
    """Caplets / floorlets (one product per payoff).

    # C++ parity: multistepoptionlets.hpp MultiStepOptionlets.
    """

    def __init__(
        self,
        rate_times: list[float],
        accruals: list[float],
        payment_times: list[float],
        payoffs: list[Payoff],
    ) -> None:
        super().__init__(rate_times)
        self._accruals = list(accruals)
        self._payment_times = list(payment_times)
        self._payoffs = list(payoffs)
        self._current_index = 0
        check_increasing_times(payment_times)

    def possible_cash_flow_times(self) -> list[float]:
        return self._payment_times

    def number_of_products(self) -> int:
        return len(self._payoffs)

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
        libor_rate = current_state.forward_rate(self._current_index)
        cf = cash_flows_generated[self._current_index][0]
        cf.time_index = self._current_index
        cf.amount = (
            self._payoffs[self._current_index](libor_rate)
            * self._accruals[self._current_index]
        )
        for i in range(len(number_cash_flows_this_step)):
            number_cash_flows_this_step[i] = 0
        number_cash_flows_this_step[self._current_index] = 1
        self._current_index += 1
        return self._current_index == len(self._payoffs)

    def clone(self) -> MarketModelMultiProduct:
        return MultiStepOptionlets(
            self._rate_times, self._accruals, self._payment_times, self._payoffs
        )
