"""MarketModelPathwiseCashRebate — pathwise (Greeks-aware) cash rebate.

# C++ parity:
# ql/models/marketmodels/products/pathwise/pathwiseproductcashrebate.{hpp,cpp}
# (v1.42.1).

A fixed-amount rebate (the pathwise analogue of ``MarketModelCashRebate``):
at the single step every product ``p`` pays ``amounts[p][currentIndex]`` with
all forward-rate derivatives zero. Used as the default rebate inside
``CallSpecifiedPathwiseMultiProduct``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.pathwise_multi_product import (
    MarketModelPathwiseMultiProduct,
    PathwiseCashFlow,
)
from pquantlib.models.marketmodels.utilities import check_increasing_times

if TYPE_CHECKING:
    from pquantlib.math.matrix import Matrix
    from pquantlib.models.marketmodels.curve_state import CurveState


class MarketModelPathwiseCashRebate(MarketModelPathwiseMultiProduct):
    """A pathwise fixed-amount cash rebate.

    # C++ parity: pathwiseproductcashrebate.hpp MarketModelPathwiseCashRebate.
    """

    def __init__(
        self,
        evolution: EvolutionDescription,
        payment_times: list[float],
        amounts: Matrix,
        number_of_products: int,
    ) -> None:
        self._evolution = evolution
        self._payment_times = list(payment_times)
        self._amounts = amounts
        self._number_of_products = number_of_products
        self._current_index = 0
        check_increasing_times(payment_times)
        qassert.require(
            amounts.shape[0] == number_of_products,
            "the number of rows in the matrix must equal the number of products",
        )
        qassert.require(
            amounts.shape[1] == len(self._payment_times),
            "the number of columns in the matrix must equal the number of payment times",
        )
        qassert.require(
            len(evolution.evolution_times()) == len(self._payment_times),
            "the number of evolution times must equal the number of payment times",
        )

    def already_deflated(self) -> bool:
        return False

    def suggested_numeraires(self) -> list[int]:
        return qassert.fail("not implemented (yet?)")

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def possible_cash_flow_times(self) -> list[float]:
        return self._payment_times

    def number_of_products(self) -> int:
        return self._number_of_products

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
        for i in range(self._number_of_products):
            number_cash_flows_this_step[i] = 1
            cf = cash_flows_generated[i][0]
            cf.time_index = self._current_index
            cf.amount[0] = float(self._amounts[i, self._current_index])
            for k in range(1, self._evolution.number_of_rates() + 1):
                cf.amount[k] = 0.0
        self._current_index += 1
        return True

    def clone(self) -> MarketModelPathwiseMultiProduct:
        return MarketModelPathwiseCashRebate(
            self._evolution, self._payment_times, self._amounts, self._number_of_products
        )
