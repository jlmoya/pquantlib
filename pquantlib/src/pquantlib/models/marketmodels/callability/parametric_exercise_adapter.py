"""ParametricExerciseAdapter — adapt a parametric exercise to a strategy.

# C++ parity:
# ql/models/marketmodels/callability/parametricexerciseadapter.{hpp,cpp}
# (v1.42.1).

Wraps a ``MarketModelParametricExercise`` plus a calibrated parameter set into
an ``ExerciseStrategy``: at each exercise time it reads the node variables from
the wrapped exercise and applies the parametric rule with the stored
parameters.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from pquantlib.models.marketmodels.callability.exercise_strategy import ExerciseStrategy

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.callability.market_model_parametric_exercise import (
        MarketModelParametricExercise,
    )
    from pquantlib.models.marketmodels.curve_state import CurveState


class ParametricExerciseAdapter(ExerciseStrategy):
    """Adapt a parametric exercise + parameters to an exercise strategy.

    # C++ parity: parametricexerciseadapter.hpp ParametricExerciseAdapter.
    """

    def __init__(
        self,
        exercise: MarketModelParametricExercise,
        parameters: list[list[float]],
    ) -> None:
        self._exercise = exercise.clone()
        self._parameters = [list(p) for p in parameters]
        self._is_exercise_time = list(exercise.is_exercise_time())
        self._number_of_variables = list(exercise.number_of_variables())
        self._current_step = 0
        self._current_exercise = 0
        self._variables: list[float] = []

        evolution_times = self._exercise.evolution().evolution_times()
        self._exercise_times = [
            evolution_times[i]
            for i in range(len(evolution_times))
            if self._is_exercise_time[i]
        ]

    def exercise_times(self) -> list[float]:
        return self._exercise_times

    def relevant_times(self) -> list[float]:
        return self._exercise.evolution().evolution_times()

    def reset(self) -> None:
        self._exercise.reset()
        self._current_step = 0
        self._current_exercise = 0

    def next_step(self, current_state: CurveState) -> None:
        self._exercise.next_step(current_state)
        if self._is_exercise_time[self._current_step]:
            self._current_exercise += 1
        self._current_step += 1

    def exercise(self, current_state: CurveState) -> bool:
        self._variables = [0.0] * self._number_of_variables[self._current_exercise - 1]
        self._exercise.values(current_state, self._variables)
        return self._exercise.exercise(
            self._current_exercise - 1,
            self._parameters[self._current_exercise - 1],
            self._variables,
        )

    def clone(self) -> ParametricExerciseAdapter:
        return copy.deepcopy(self)
