"""MarketModelCashRebate — one-shot fixed cash amount product.

# C++ parity: ql/models/marketmodels/products/multistep/cashrebate.{hpp,cpp}
# (v1.42.1).

Models receipt of a fixed cash amount once; the product terminates immediately.
Mainly useful as the rebate received when another product is cancelled (see
``CallSpecifiedMultiProduct``). ``amounts`` is a ``(numberOfProducts by
len(paymentTimes))`` matrix: at step ``currentIndex``, product ``i`` receives
``amounts[i][currentIndex]`` at cash-flow-time index ``currentIndex``.
"""

from __future__ import annotations

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix
from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.multi_product import (
    CashFlow,
    MarketModelMultiProduct,
)
from pquantlib.models.marketmodels.utilities import check_increasing_times


class MarketModelCashRebate(MarketModelMultiProduct):
    """A product paying a fixed cash amount once, then terminating.

    # C++ parity: cashrebate.hpp MarketModelCashRebate.
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
        self._amounts = np.asarray(amounts, dtype=np.float64)
        self._number_of_products = number_of_products
        self._current_index = 0

        check_increasing_times(self._payment_times)
        qassert.require(
            self._amounts.shape[0] == number_of_products,
            "the number of rows in the matrix must equal the number of products",
        )
        qassert.require(
            self._amounts.shape[1] == len(self._payment_times),
            "the number of columns in the matrix must equal "
            "the number of payment times",
        )
        qassert.require(
            len(evolution.evolution_times()) == len(self._payment_times),
            "the number of evolution times must equal the number of payment times",
        )

    def possible_cash_flow_times(self) -> list[float]:
        return self._payment_times

    def number_of_products(self) -> int:
        return self._number_of_products

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return 1

    def reset(self) -> None:
        self._current_index = 0

    def suggested_numeraires(self) -> list[int]:
        qassert.fail("not implemented (yet?)")

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        for i in range(self._number_of_products):
            number_cash_flows_this_step[i] = 1
            cash_flows_generated[i][0].time_index = self._current_index
            cash_flows_generated[i][0].amount = float(
                self._amounts[i][self._current_index]
            )
        self._current_index += 1
        return True

    def clone(self) -> MarketModelMultiProduct:
        return MarketModelCashRebate(
            self._evolution,
            self._payment_times,
            self._amounts.copy(),
            self._number_of_products,
        )
