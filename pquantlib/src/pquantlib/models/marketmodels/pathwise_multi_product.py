"""MarketModelPathwiseMultiProduct — pathwise (Greeks-aware) product base.

# C++ parity: ql/models/marketmodels/pathwisemultiproduct.hpp (v1.42.1).

Like ``MarketModelMultiProduct``, but each cash flow also carries the
derivative of the pay-off with respect to each forward rate (so the
``CashFlow.amount`` is a *vector*). ``already_deflated()`` reports whether
the product already discounts its own cash flows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.curve_state import CurveState
    from pquantlib.models.marketmodels.evolution_description import EvolutionDescription


def _empty_float_list() -> list[float]:
    return []


@dataclass(slots=True)
class PathwiseCashFlow:
    """A pathwise cash flow: amount-plus-derivatives at a time index.

    # C++ parity: pathwisemultiproduct.hpp
    MarketModelPathwiseMultiProduct::CashFlow.

    ``amount`` is a vector: ``amount[0]`` is the cash-flow value and
    ``amount[1:]`` its derivatives w.r.t. each forward rate.
    """

    time_index: int = 0
    amount: list[float] = field(default_factory=_empty_float_list)


class MarketModelPathwiseMultiProduct(ABC):
    """Abstract base for pathwise (Greeks-aware) market-model products.

    # C++ parity: pathwisemultiproduct.hpp MarketModelPathwiseMultiProduct.
    """

    #: Inner cash-flow type (mirrors the C++ nested ``struct CashFlow``).
    CashFlow = PathwiseCashFlow

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
    def already_deflated(self) -> bool:
        """Whether the product already discounts its own cash flows."""

    @abstractmethod
    def reset(self) -> None:
        """Put the product at the start of a path (called before simulation)."""

    @abstractmethod
    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[PathwiseCashFlow]],
    ) -> bool:
        """Generate this step's cash flows; return ``True`` when the path is done.

        ``number_cash_flows_this_step`` and ``cash_flows_generated`` are
        caller-supplied output buffers (C++ out-parameters), filled in place.
        """

    @abstractmethod
    def clone(self) -> MarketModelPathwiseMultiProduct:
        """A newly-allocated copy of this product."""
