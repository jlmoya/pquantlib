"""ExerciseStrategy — abstract early-exercise decision rule over a CurveState.

# C++ parity: ql/methods/montecarlo/exercisestrategy.hpp
# ExerciseStrategy<CurveState> (v1.42.1).

The decision rule a callable product consults: at each exercise time it answers
"exercise now?" given the current ``CurveState``. Concretes
(``SwapRateTrigger``, ``ParametricExerciseAdapter``,
``LongstaffSchwartzExerciseStrategy``) implement the actual logic.

This concrete ABC is the structural twin of the forward-declared
``ExerciseStrategy`` Protocol exported from
``pquantlib.models.marketmodels.products`` (W11-A); both describe the same C++
``ExerciseStrategy<CurveState>`` interface, so any subclass here also satisfies
the W11-A Protocol consumed by ``CallSpecifiedMultiProduct``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.curve_state import CurveState


class ExerciseStrategy(ABC):
    """Abstract base for market-model exercise strategies.

    # C++ parity: exercisestrategy.hpp ExerciseStrategy<CurveState>.
    """

    @abstractmethod
    def exercise_times(self) -> list[float]:
        """The times at which an exercise decision is taken."""

    @abstractmethod
    def relevant_times(self) -> list[float]:
        """All times at which the strategy must observe the state."""

    @abstractmethod
    def reset(self) -> None:
        """Reset to the start of a path."""

    @abstractmethod
    def exercise(self, current_state: CurveState) -> bool:
        """Whether to exercise at the current step."""

    @abstractmethod
    def next_step(self, current_state: CurveState) -> None:
        """Advance to the next evolution step, updating internal state."""

    @abstractmethod
    def clone(self) -> ExerciseStrategy:
        """A newly-allocated copy of this strategy."""
