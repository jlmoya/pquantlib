"""MarketModelExerciseValue — abstract value-received-on-exercise object.

# C++ parity: ql/models/marketmodels/callability/exercisevalue.hpp (v1.42.1).

The callability "what do I get if I exercise now" object. Concretes
(``NothingExerciseValue``, ``BermudanSwaptionExerciseValue``) drive the LS
node-data collection and the calibrated LS exercise strategy.

This concrete ABC is the structural twin of the forward-declared
``MarketModelExerciseValue`` Protocol exported from
``pquantlib.models.marketmodels.products`` (W11-A); both describe the same C++
``MarketModelExerciseValue`` interface, so any subclass here also satisfies the
W11-A Protocol (used by ``ExerciseAdapter`` / ``CallSpecifiedMultiProduct``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.curve_state import CurveState
    from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
    from pquantlib.models.marketmodels.multi_product import CashFlow


class MarketModelExerciseValue(ABC):
    """Abstract base for market-model exercise values.

    # C++ parity: exercisevalue.hpp MarketModelExerciseValue.
    """

    @abstractmethod
    def number_of_exercises(self) -> int:
        """The number of exercise opportunities."""

    @abstractmethod
    def evolution(self) -> EvolutionDescription:
        """The evolution description (including state-update times)."""

    @abstractmethod
    def possible_cash_flow_times(self) -> list[float]:
        """All times at which an exercise cash flow could be paid."""

    @abstractmethod
    def next_step(self, current_state: CurveState) -> None:
        """Advance to the next evolution step, updating internal state."""

    @abstractmethod
    def reset(self) -> None:
        """Reset to the start of a path."""

    @abstractmethod
    def is_exercise_time(self) -> list[bool]:
        """Per evolution time, whether it is an exercise time."""

    @abstractmethod
    def value(self, current_state: CurveState) -> CashFlow:
        """The cash flow received on exercising at the current step."""

    @abstractmethod
    def clone(self) -> MarketModelExerciseValue:
        """A newly-allocated copy of this exercise value."""
