"""MultiProductMultiStep — abstract base for multiple-step BGM products.

# C++ parity: ql/models/marketmodels/products/multiproductmultistep.{hpp,cpp}
# (v1.42.1).

A ``MarketModelMultiProduct`` evaluated in more than one step (Rebonato's
"long jump"). Builds the canonical per-rate-time ``EvolutionDescription`` and a
MoneyMarketPlus(1) suggested-numeraire vector. Concrete subclasses implement
the remaining ``MarketModelMultiProduct`` interface.
"""

from __future__ import annotations

from abc import ABC

from pquantlib import qassert
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.multi_product import MarketModelMultiProduct


class MultiProductMultiStep(MarketModelMultiProduct, ABC):
    """Abstract multiple-step market-model product.

    # C++ parity: multiproductmultistep.hpp MultiProductMultiStep.
    """

    def __init__(self, rate_times: list[float]) -> None:
        qassert.require(
            len(rate_times) > 1, "Rate times must contain at least two values"
        )
        self._rate_times = list(rate_times)
        n = len(self._rate_times) - 1
        evolution_times = [self._rate_times[i] for i in range(n)]
        relevance_rates = [(i, i + 1) for i in range(n)]
        self._evolution = EvolutionDescription(
            self._rate_times, evolution_times, relevance_rates
        )

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def suggested_numeraires(self) -> list[int]:
        # MoneyMarketPlus(1): numeraire bond i+1 at step i.
        n = len(self._rate_times) - 1
        return [i + 1 for i in range(n)]
