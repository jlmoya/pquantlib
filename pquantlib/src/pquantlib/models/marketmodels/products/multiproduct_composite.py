"""MultiProductComposite — composite exposing the union of sub-products.

# C++ parity: ql/models/marketmodels/products/multiproductcomposite.{hpp,cpp}
# (v1.42.1).

The composite's ``number_of_products`` is the sum of the sub-products'; each
sub-product's outputs are laid into the global buffers at its product offset,
with time indices remapped into the merged cash-flow-time vector and amounts
scaled by the sub-product multiplier.
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


class MultiProductComposite(MarketModelComposite):
    """Composition of one or more market-model products (union semantics).

    # C++ parity: multiproductcomposite.hpp MultiProductComposite.
    """

    def number_of_products(self) -> int:
        return sum(c.product.number_of_products() for c in self._components)

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return max(
            (
                c.product.max_number_of_cash_flows_per_product_per_step()
                for c in self._components
            ),
            default=0,
        )

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        qassert.require(self._finalized, "composite not finalized")
        done = True
        offset = 0
        for n, sub in enumerate(self._components):
            n_products = sub.product.number_of_products()
            if self._is_in_subset[n][self._current_index] and not sub.done:
                this_done = sub.product.next_time_step(
                    current_state, sub.number_of_cashflows, sub.cashflows
                )
                for j in range(n_products):
                    number_cash_flows_this_step[j + offset] = sub.number_of_cashflows[j]
                    for k in range(sub.number_of_cashflows[j]):
                        src = sub.cashflows[j][k]
                        dst = cash_flows_generated[j + offset][k]
                        dst.time_index = sub.time_indices[src.time_index]
                        dst.amount = src.amount * sub.multiplier
                done = done and this_done
            else:
                for j in range(n_products):
                    number_cash_flows_this_step[j + offset] = 0
            offset += n_products
        self._current_index += 1
        return done

    def clone(self) -> MarketModelMultiProduct:
        return self._copy_into(MultiProductComposite())
