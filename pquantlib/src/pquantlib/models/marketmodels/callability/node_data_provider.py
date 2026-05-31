"""MarketModelNodeDataProvider — abstract per-exercise data supplier.

# C++ parity: ql/models/marketmodels/callability/nodedataprovider.hpp
# (v1.42.1).

The common base of ``MarketModelBasisSystem`` (LS regressors) and
``MarketModelParametricExercise`` (parametric variables): for each exercise it
supplies a vector of ``Real`` data computed from the current ``CurveState``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.curve_state import CurveState
    from pquantlib.models.marketmodels.evolution_description import EvolutionDescription


class MarketModelNodeDataProvider(ABC):
    """Abstract per-exercise node-data supplier.

    # C++ parity: nodedataprovider.hpp MarketModelNodeDataProvider.
    """

    @abstractmethod
    def number_of_exercises(self) -> int:
        """The number of exercise opportunities."""

    @abstractmethod
    def number_of_data(self) -> list[int]:
        """The number of data values supplied per exercise."""

    @abstractmethod
    def evolution(self) -> EvolutionDescription:
        """The evolution description (including state-update times)."""

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
    def values(self, current_state: CurveState, results: list[float]) -> None:
        """Fill ``results`` with the node data for the current exercise.

        ``results`` is a caller-supplied output list (C++ out-parameter),
        resized/overwritten in place.
        """
