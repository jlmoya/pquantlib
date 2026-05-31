"""MarketModelComposite — abstract composition of BGM sub-products.

# C++ parity: ql/models/marketmodels/products/compositeproduct.{hpp,cpp}
# (v1.42.1).

Builds a market-model product by composing one or more sub-products that all
share the same rate times. ``add`` / ``subtract`` register sub-products (each
with a multiplier); ``finalize`` merges their evolution times and the union of
their cash-flow times, after which the composite is a usable
``MarketModelMultiProduct``. The two concrete leaves (``MultiProductComposite``,
``SingleProductComposite``) only differ in how ``next_time_step`` lays the
sub-product cash flows into the output buffers.

Divergences from C++:

- C++ stores each sub-product as a ``Clone<MarketModelMultiProduct>`` (deep copy
  on ``add``). The Python port calls ``product.clone()`` on ``add`` for the same
  single-ownership / value semantics.
- C++ ``mergeTimes`` fills an out-param ``isInSubset_``; the Python
  ``merge_times`` returns ``(merged, is_in_subset)`` directly.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from pquantlib import qassert
from pquantlib.models.marketmodels.evolution_description import (
    EvolutionDescription,
    terminal_measure,
)
from pquantlib.models.marketmodels.multi_product import (
    CashFlow,
    MarketModelMultiProduct,
)
from pquantlib.models.marketmodels.utilities import merge_times


@dataclass(slots=True)
class _SubProduct:
    """One registered sub-product + its per-step working buffers.

    # C++ parity: compositeproduct.hpp MarketModelComposite::SubProduct.
    """

    product: MarketModelMultiProduct
    multiplier: float
    done: bool = False
    number_of_cashflows: list[int] = field(default_factory=list[int])
    cashflows: list[list[CashFlow]] = field(default_factory=list[list[CashFlow]])
    time_indices: list[int] = field(default_factory=list[int])


class MarketModelComposite(MarketModelMultiProduct, ABC):
    """Abstract composite of two or more market-model products.

    # C++ parity: compositeproduct.hpp MarketModelComposite.
    """

    def __init__(self) -> None:
        self._components: list[_SubProduct] = []
        self._rate_times: list[float] = []
        self._evolution_times: list[float] = []
        self._evolution: EvolutionDescription | None = None
        self._finalized = False
        self._current_index = 0
        self._cashflow_times: list[float] = []
        self._all_evolution_times: list[list[float]] = []
        self._is_in_subset: list[list[bool]] = []

    # --- MarketModelMultiProduct interface ----------------------------------
    def evolution(self) -> EvolutionDescription:
        qassert.require(self._finalized, "composite not finalized")
        assert self._evolution is not None
        return self._evolution

    def suggested_numeraires(self) -> list[int]:
        qassert.require(self._finalized, "composite not finalized")
        assert self._evolution is not None
        return terminal_measure(self._evolution)

    def possible_cash_flow_times(self) -> list[float]:
        qassert.require(self._finalized, "composite not finalized")
        return self._cashflow_times

    def reset(self) -> None:
        for component in self._components:
            component.product.reset()
            component.done = False
        self._current_index = 0

    # --- composite facilities -----------------------------------------------
    def add(self, product: MarketModelMultiProduct, multiplier: float = 1.0) -> None:
        qassert.require(not self._finalized, "product already finalized")
        d = product.evolution()
        if self._components:
            rate_times1 = self._components[0].product.evolution().rate_times()
            rate_times2 = d.rate_times()
            qassert.require(
                len(rate_times1) == len(rate_times2)
                and all(a == b for a, b in zip(rate_times1, rate_times2, strict=False)),
                "incompatible rate times",
            )
        self._components.append(_SubProduct(product=product.clone(), multiplier=multiplier))
        self._all_evolution_times.append(list(d.evolution_times()))

    def subtract(
        self, product: MarketModelMultiProduct, multiplier: float = 1.0
    ) -> None:
        self.add(product, -multiplier)

    def finalize(self) -> None:
        qassert.require(not self._finalized, "product already finalized")
        qassert.require(bool(self._components), "no sub-product provided")

        description = self._components[0].product.evolution()
        self._rate_times = list(description.rate_times())

        self._evolution_times, self._is_in_subset = merge_times(
            self._all_evolution_times
        )

        all_cashflow_times: list[float] = []
        for sub in self._components:
            cashflow_times = sub.product.possible_cash_flow_times()
            all_cashflow_times.extend(cashflow_times)
            n_products = sub.product.number_of_products()
            sub.number_of_cashflows = [0] * n_products
            max_flows = sub.product.max_number_of_cash_flows_per_product_per_step()
            sub.cashflows = [
                [CashFlow() for _ in range(max_flows)] for _ in range(n_products)
            ]

        # sort + unique the union of all cash-flow times
        self._cashflow_times = sorted(set(all_cashflow_times))

        # map each sub-product's cash-flow time into the merged vector
        for sub in self._components:
            product_times = sub.product.possible_cash_flow_times()
            sub.time_indices = [
                self._cashflow_times.index(t) for t in product_times
            ]

        self._evolution = EvolutionDescription(
            self._rate_times, self._evolution_times
        )
        self._finalized = True

    def _copy_into(self, target: MarketModelComposite) -> MarketModelComposite:
        """Re-add + (re-)finalize this composite's components into ``target``.

        # C++ parity: the copy-ctor deep-copies every SubProduct via Clone<>. The
        # Python port re-adds each component (which clones the sub-product) and
        # re-finalizes, reproducing the post-finalize state with independent
        # sub-products. Used by the concrete leaves' ``clone()``.
        """
        for c in self._components:
            target.add(c.product, c.multiplier)
        if self._finalized:
            target.finalize()
        return target

    def size(self) -> int:
        return len(self._components)

    def item(self, i: int) -> MarketModelMultiProduct:
        return self._components[i].product

    def multiplier(self, i: int) -> float:
        return self._components[i].multiplier
