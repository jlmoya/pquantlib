"""SingleProductComposite — composite collapsing sub-products into one product.

# C++ parity: ql/models/marketmodels/products/singleproductcomposite.{hpp,cpp}
# (v1.42.1).

The composite's ``number_of_products`` is 1; every sub-product's cash flows are
concatenated into product 0's buffer (running offset across sub-products and
their products), with time indices remapped and amounts scaled by the
sub-product multiplier.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.multi_product import (
    CashFlow,
    MarketModelMultiProduct,
)
from pquantlib.models.marketmodels.products.composite_product import (
    MarketModelComposite,
)


class SingleProductComposite(MarketModelComposite):
    """Composition of one or more products collapsed into a single product.

    # C++ parity: singleproductcomposite.hpp SingleProductComposite.
    """

    def number_of_products(self) -> int:
        return 1

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return sum(
            c.product.max_number_of_cash_flows_per_product_per_step()
            for c in self._components
        )

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        qassert.require(self._finalized, "composite not finalized")
        done = True
        total_cashflows = 0
        for n, sub in enumerate(self._components):
            if self._is_in_subset[n][self._current_index] and not sub.done:
                this_done = sub.product.next_time_step(
                    current_state, sub.number_of_cashflows, sub.cashflows
                )
                for j in range(sub.product.number_of_products()):
                    offset = total_cashflows
                    total_cashflows += sub.number_of_cashflows[j]
                    for k in range(sub.number_of_cashflows[j]):
                        src = sub.cashflows[j][k]
                        dst = cash_flows_generated[0][k + offset]
                        dst.time_index = sub.time_indices[src.time_index]
                        dst.amount = src.amount * sub.multiplier
                    number_cash_flows_this_step[0] = total_cashflows
                done = done and this_done
        self._current_index += 1
        return done

    def clone(self) -> MarketModelMultiProduct:
        return self._copy_into(SingleProductComposite())
