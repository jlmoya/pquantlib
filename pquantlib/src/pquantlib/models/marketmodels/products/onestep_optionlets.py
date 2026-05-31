"""OneStepOptionlets — caplets / floorlets as a single-step product.

# C++ parity: ql/models/marketmodels/products/onestep/onestepoptionlets.{hpp,cpp}
# (v1.42.1).

One product per payoff. In the single step, each product ``i`` whose
``payoff[i](forwardRate(i))`` is strictly positive pays
``payoff[i](forwardRate(i)) * accrual[i]`` at payment-time index ``i`` (out-of-
the-money products generate no cash flow).
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
from pquantlib.payoffs import Payoff


class OneStepOptionlets(MultiProductOneStep):
    """Caplets / floorlets evaluated in a single step.

    # C++ parity: onestepoptionlets.hpp OneStepOptionlets.
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
        check_increasing_times(payment_times)

    def possible_cash_flow_times(self) -> list[float]:
        return self._payment_times

    def number_of_products(self) -> int:
        return len(self._payoffs)

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return 1

    def reset(self) -> None:
        # nothing to do
        pass

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        for i in range(len(number_cash_flows_this_step)):
            number_cash_flows_this_step[i] = 0
        for i in range(len(self._payoffs)):
            libor_rate = current_state.forward_rate(i)
            payoff = self._payoffs[i](libor_rate)
            if payoff > 0.0:
                number_cash_flows_this_step[i] = 1
                cf = cash_flows_generated[i][0]
                cf.time_index = i
                cf.amount = payoff * self._accruals[i]
        return True

    def clone(self) -> MarketModelMultiProduct:
        return OneStepOptionlets(
            self._rate_times, self._accruals, self._payment_times, self._payoffs
        )
