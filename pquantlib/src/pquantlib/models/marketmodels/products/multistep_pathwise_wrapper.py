"""MultiProductPathwiseWrapper — treat a pathwise product as an ordinary one.

# C++ parity:
# ql/models/marketmodels/products/multistep/multisteppathwisewrapper.{hpp,cpp}
# (v1.42.1).

Pathwise products do everything ordinary products do and more, so a pathwise
product can be wrapped as an ordinary ``MarketModelMultiProduct`` (discarding
the Greeks). This lets a product be written once. The wrapper drives the inner
pathwise product into its own scratch buffers and copies only ``amount[0]`` (the
cash-flow value) into the ordinary cash flows. Tested in
MarketModels::testInverseFloater.
"""

from __future__ import annotations

from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.multi_product import (
    CashFlow,
    MarketModelMultiProduct,
)
from pquantlib.models.marketmodels.pathwise_multi_product import (
    MarketModelPathwiseMultiProduct,
    PathwiseCashFlow,
)


class MultiProductPathwiseWrapper(MarketModelMultiProduct):
    """Wrap a pathwise product as an ordinary (non-Greeks) product.

    # C++ parity: multisteppathwisewrapper.hpp MultiProductPathwiseWrapper.
    """

    def __init__(self, inner_product: MarketModelPathwiseMultiProduct) -> None:
        self._inner_product = inner_product.clone()
        self._number_of_products = inner_product.number_of_products()
        max_cf = inner_product.max_number_of_cash_flows_per_product_per_step()
        n_rates = inner_product.evolution().number_of_rates()
        self._cash_flows_generated: list[list[PathwiseCashFlow]] = [
            [PathwiseCashFlow(amount=[0.0] * (1 + n_rates)) for _ in range(max_cf)]
            for _ in range(self._number_of_products)
        ]

    def suggested_numeraires(self) -> list[int]:
        return self._inner_product.suggested_numeraires()

    def evolution(self) -> EvolutionDescription:
        return self._inner_product.evolution()

    def possible_cash_flow_times(self) -> list[float]:
        return self._inner_product.possible_cash_flow_times()

    def number_of_products(self) -> int:
        return self._inner_product.number_of_products()

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return self._inner_product.max_number_of_cash_flows_per_product_per_step()

    def reset(self) -> None:
        self._inner_product.reset()

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        done = self._inner_product.next_time_step(
            current_state, number_cash_flows_this_step, self._cash_flows_generated
        )
        # transform the data: keep only amount[0].
        for i in range(self._number_of_products):
            for j in range(number_cash_flows_this_step[i]):
                cash_flows_generated[i][j].time_index = self._cash_flows_generated[i][
                    j
                ].time_index
                cash_flows_generated[i][j].amount = self._cash_flows_generated[i][j].amount[0]
        return done

    def clone(self) -> MarketModelMultiProduct:
        return MultiProductPathwiseWrapper(self._inner_product)
