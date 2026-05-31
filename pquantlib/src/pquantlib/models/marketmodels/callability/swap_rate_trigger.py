"""SwapRateTrigger — exercise when the coterminal swap rate beats a trigger.

# C++ parity: ql/models/marketmodels/callability/swapratetrigger.{hpp,cpp}
# (v1.42.1).

A naive exercise strategy: at exercise opportunity ``k`` (with associated rate
index ``rateIndex[k]``) it exercises iff ``swapTriggers[k] < coterminalSwapRate(
rateIndex[k])``.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.models.marketmodels.callability.exercise_strategy import ExerciseStrategy
from pquantlib.models.marketmodels.utilities import check_increasing_times

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.curve_state import CurveState


class SwapRateTrigger(ExerciseStrategy):
    """Exercise when the coterminal swap rate exceeds a per-exercise trigger.

    # C++ parity: swapratetrigger.hpp SwapRateTrigger.
    """

    def __init__(
        self,
        rate_times: list[float],
        swap_triggers: list[float],
        exercise_times: list[float],
    ) -> None:
        check_increasing_times(rate_times)
        qassert.require(
            len(rate_times) > 1, "Rate times must contain at least two values"
        )
        check_increasing_times(exercise_times)
        qassert.require(
            len(swap_triggers) == len(exercise_times),
            "swapTriggers/exerciseTimes mismatch",
        )
        self._rate_times = list(rate_times)
        self._swap_triggers = list(swap_triggers)
        self._exercise_times = list(exercise_times)
        self._current_index = 0

        self._rate_index = [0] * len(exercise_times)
        j = 0
        for i in range(len(exercise_times)):
            while j < len(rate_times) and rate_times[j] < exercise_times[i]:
                j += 1
            self._rate_index[i] = j

    def exercise_times(self) -> list[float]:
        return self._exercise_times

    def relevant_times(self) -> list[float]:
        return self._exercise_times

    def reset(self) -> None:
        self._current_index = 0

    def exercise(self, current_state: CurveState) -> bool:
        rate_index = self._rate_index[self._current_index - 1]
        current_swap_rate = current_state.coterminal_swap_rate(rate_index)
        return self._swap_triggers[self._current_index - 1] < current_swap_rate

    def next_step(self, current_state: CurveState) -> None:
        self._current_index += 1

    def clone(self) -> SwapRateTrigger:
        return copy.deepcopy(self)
