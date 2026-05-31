"""MultiStepNothing — a product that generates no cash flows (test scaffold).

# C++ parity: ql/models/marketmodels/products/multistep/multistepnothing.{hpp,cpp}
# (v1.42.1).

Produces no cash flows; advances ``currentIndex`` each step and reports done
once it reaches ``doneIndex``. Used as a no-op test scaffold.
"""

from __future__ import annotations

from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.multi_product import (
    CashFlow,
    MarketModelMultiProduct,
)
from pquantlib.models.marketmodels.products.multiproduct_multistep import (
    MultiProductMultiStep,
)


class MultiStepNothing(MultiProductMultiStep):
    """A product that produces no cash flows.

    # C++ parity: multistepnothing.hpp MultiStepNothing.
    """

    def __init__(
        self,
        evolution: EvolutionDescription,
        number_of_products: int = 1,
        done_index: int = 0,
    ) -> None:
        super().__init__(evolution.rate_times())
        self._number_of_products = number_of_products
        self._done_index = done_index
        self._current_index = 0

    def possible_cash_flow_times(self) -> list[float]:
        return []

    def number_of_products(self) -> int:
        return self._number_of_products

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return 0

    def reset(self) -> None:
        self._current_index = 0

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        for i in range(len(number_cash_flows_this_step)):
            number_cash_flows_this_step[i] = 0
        self._current_index += 1
        return self._current_index >= self._done_index

    def clone(self) -> MarketModelMultiProduct:
        return MultiStepNothing(
            self._evolution, self._number_of_products, self._done_index
        )
