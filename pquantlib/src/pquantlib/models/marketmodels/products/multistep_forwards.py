"""MultiStepForwards — forward-rate agreements as a multi-step product.

# C++ parity: ql/models/marketmodels/products/multistep/multistepforwards.{hpp,cpp}
# (v1.42.1).

One product per strike. At step ``i`` product ``i`` pays
``(forwardRate(i) - strike[i]) * accrual[i]`` at payment-time index ``i``.
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


class MultiStepForwards(MultiProductMultiStep):
    """Forward-rate agreements (one product per strike).

    # C++ parity: multistepforwards.hpp MultiStepForwards.
    """

    def __init__(
        self,
        rate_times: list[float],
        accruals: list[float],
        payment_times: list[float],
        strikes: list[float],
    ) -> None:
        super().__init__(rate_times)
        self._accruals = list(accruals)
        self._payment_times = list(payment_times)
        self._strikes = list(strikes)
        self._current_index = 0
        check_increasing_times(payment_times)

    def possible_cash_flow_times(self) -> list[float]:
        return self._payment_times

    def number_of_products(self) -> int:
        return len(self._strikes)

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
        cf.amount = (libor_rate - self._strikes[self._current_index]) * self._accruals[
            self._current_index
        ]
        for i in range(len(number_cash_flows_this_step)):
            number_cash_flows_this_step[i] = 0
        number_cash_flows_this_step[self._current_index] = 1
        self._current_index += 1
        return self._current_index == len(self._strikes)

    def clone(self) -> MarketModelMultiProduct:
        return MultiStepForwards(
            self._rate_times, self._accruals, self._payment_times, self._strikes
        )
