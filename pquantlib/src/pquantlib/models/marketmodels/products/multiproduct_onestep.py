"""MultiProductOneStep — abstract base for single-step BGM products.

# C++ parity: ql/models/marketmodels/products/multiproductonestep.{hpp,cpp}
# (v1.42.1).

A ``MarketModelMultiProduct`` evaluated in a single step (Rebonato's "very long
jump"). The evolution has a single evolution time at the next-to-last rate time,
a single relevance-rate range spanning all rates, and the terminal-measure
suggested numeraire.
"""

from __future__ import annotations

from abc import ABC

from pquantlib import qassert
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.multi_product import MarketModelMultiProduct


class MultiProductOneStep(MarketModelMultiProduct, ABC):
    """Abstract single-step market-model product.

    # C++ parity: multiproductonestep.hpp MultiProductOneStep.
    """

    def __init__(self, rate_times: list[float]) -> None:
        qassert.require(
            len(rate_times) > 1, "Rate times must contain at least two values"
        )
        self._rate_times = list(rate_times)
        evolution_times = [self._rate_times[len(self._rate_times) - 2]]
        relevance_rates = [(0, len(self._rate_times) - 1)]
        self._evolution = EvolutionDescription(
            self._rate_times, evolution_times, relevance_rates
        )

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def suggested_numeraires(self) -> list[int]:
        # Terminal measure.
        return [len(self._rate_times) - 1]
