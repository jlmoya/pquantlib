"""MarketModelEvolver — abstract forward-rate evolver.

# C++ parity: ql/models/marketmodels/evolver.hpp (v1.42.1).

The evolver does the actual work of evolving the forward rates from one
evolution time to the next, exposing the current ``CurveState`` after each
step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.curve_state import CurveState


class MarketModelEvolver(ABC):
    """Abstract base for market-model evolvers.

    # C++ parity: evolver.hpp MarketModelEvolver.
    """

    @abstractmethod
    def numeraires(self) -> list[int]:
        """The per-step numeraire rate indices."""

    @abstractmethod
    def start_new_path(self) -> float:
        """Reset to the start of a new path; return the path weight."""

    @abstractmethod
    def advance_step(self) -> float:
        """Advance one evolution step; return the incremental weight."""

    @abstractmethod
    def current_step(self) -> int:
        """The current evolution-step index."""

    @abstractmethod
    def current_state(self) -> CurveState:
        """The curve state after the current step."""

    @abstractmethod
    def set_initial_state(self, curve_state: CurveState) -> None:
        """Set the initial curve state for subsequent paths."""
