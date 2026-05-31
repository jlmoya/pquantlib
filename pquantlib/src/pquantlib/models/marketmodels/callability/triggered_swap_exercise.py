"""TriggeredSwapExercise — parametric swap-rate-trigger exercise.

# C++ parity:
# ql/models/marketmodels/callability/triggeredswapexercise.{hpp,cpp} (v1.42.1).

A ``MarketModelParametricExercise`` whose single node variable is the coterminal
swap rate and whose single parameter is a trigger level: exercise iff
``variables[0] >= parameters[0]``. The initial guess for the trigger is the
per-exercise strike.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from pquantlib.models.marketmodels.callability.market_model_parametric_exercise import (
    MarketModelParametricExercise,
)
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.curve_state import CurveState


class TriggeredSwapExercise(MarketModelParametricExercise):
    """Parametric swap-rate-trigger exercise.

    # C++ parity: triggeredswapexercise.hpp TriggeredSwapExercise.
    """

    def __init__(
        self,
        rate_times: list[float],
        exercise_times: list[float],
        strikes: list[float],
    ) -> None:
        self._rate_times = list(rate_times)
        self._exercise_times = list(exercise_times)
        self._strikes = list(strikes)
        self._current_step = 0
        self._evolution = EvolutionDescription(rate_times, exercise_times)

        self._rate_index = [0] * len(exercise_times)
        j = 0
        for i in range(len(exercise_times)):
            while j < len(rate_times) and rate_times[j] < exercise_times[i]:
                j += 1
            self._rate_index[i] = j

    # --- NodeDataProvider interface -----------------------------------------
    def number_of_exercises(self) -> int:
        return len(self._exercise_times)

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def next_step(self, current_state: CurveState) -> None:
        self._current_step += 1

    def reset(self) -> None:
        self._current_step = 0

    def is_exercise_time(self) -> list[bool]:
        return [True] * self.number_of_exercises()

    def values(self, current_state: CurveState, results: list[float]) -> None:
        swap_index = self._rate_index[self._current_step - 1]
        results[:] = [current_state.coterminal_swap_rate(swap_index)]

    # --- ParametricExercise interface ---------------------------------------
    def number_of_variables(self) -> list[int]:
        return [1] * self.number_of_exercises()

    def number_of_parameters(self) -> list[int]:
        return [1] * self.number_of_exercises()

    def exercise(
        self,
        exercise_number: int,
        parameters: list[float],
        variables: list[float],
    ) -> bool:
        return variables[0] >= parameters[0]

    def guess(self, exercise_number: int, parameters: list[float]) -> None:
        parameters[:] = [self._strikes[exercise_number]]

    def clone(self) -> TriggeredSwapExercise:
        return copy.deepcopy(self)
