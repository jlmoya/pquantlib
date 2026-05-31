"""MarketModelMultiProduct — abstract product termsheet + cash-flow generator.

# C++ parity: ql/models/marketmodels/multiproduct.hpp (v1.42.1).

Encapsulates the termsheet of a (multi-)product: for each evolution time it
generates the cash flows associated with that time for the current
``CurveState``. A callable product would fold its exercise strategy in here.

The nested ``CashFlow`` dataclass mirrors the C++ ``struct CashFlow`` (a
``timeIndex`` into ``possible_cash_flow_times()`` + an ``amount``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.curve_state import CurveState
    from pquantlib.models.marketmodels.evolution_description import EvolutionDescription


@dataclass(slots=True)
class CashFlow:
    """A generated cash flow: amount at a possible-cash-flow-time index.

    # C++ parity: multiproduct.hpp MarketModelMultiProduct::CashFlow.

    Mutable (not frozen) because evolvers reuse a preallocated pool of
    ``CashFlow`` objects across steps, overwriting ``time_index`` / ``amount``
    in place — matching the C++ ``std::vector<std::vector<CashFlow>>`` reuse.
    """

    time_index: int = 0
    amount: float = 0.0


class MarketModelMultiProduct(ABC):
    """Abstract base for market-model (multi-)products.

    # C++ parity: multiproduct.hpp MarketModelMultiProduct.
    """

    #: Inner cash-flow type (mirrors the C++ nested ``struct CashFlow``).
    CashFlow = CashFlow

    @abstractmethod
    def suggested_numeraires(self) -> list[int]:
        """A natural numeraire choice for this product."""

    @abstractmethod
    def evolution(self) -> EvolutionDescription:
        """The evolution description this product is defined against."""

    @abstractmethod
    def possible_cash_flow_times(self) -> list[float]:
        """All times at which a cash flow could be generated."""

    @abstractmethod
    def number_of_products(self) -> int:
        """The number of (sub-)products."""

    @abstractmethod
    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        """Upper bound on cash flows per product per evolution step."""

    @abstractmethod
    def reset(self) -> None:
        """Put the product at the start of a path (called before simulation)."""

    @abstractmethod
    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        """Generate this step's cash flows; return ``True`` when the path is done.

        ``number_cash_flows_this_step`` and ``cash_flows_generated`` are
        caller-supplied output buffers (C++ out-parameters), filled in place.
        """

    @abstractmethod
    def clone(self) -> MarketModelMultiProduct:
        """A newly-allocated copy of this product."""
