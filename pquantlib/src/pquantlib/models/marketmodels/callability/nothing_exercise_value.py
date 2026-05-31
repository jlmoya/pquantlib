"""NothingExerciseValue — an exercise value that always pays zero.

# C++ parity: ql/models/marketmodels/callability/nothingexercisevalue.{hpp,cpp}
# (v1.42.1).

Used as the ``control`` and the ``nullRebate`` in the Longstaff-Schwartz
exercise-strategy calibration: it has the right evolution / exercise-time
geometry but its cash flow is always ``0``.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.models.marketmodels.callability.exercise_value import (
    MarketModelExerciseValue,
)
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.multi_product import CashFlow
from pquantlib.models.marketmodels.utilities import check_increasing_times

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.curve_state import CurveState


class NothingExerciseValue(MarketModelExerciseValue):
    """An exercise value that always returns a zero cash flow.

    # C++ parity: nothingexercisevalue.hpp NothingExerciseValue.
    """

    def __init__(
        self, rate_times: list[float], is_exercise_time: list[bool] | None = None
    ) -> None:
        check_increasing_times(rate_times)
        qassert.require(
            len(rate_times) >= 2, "Rate times must contain at least two values"
        )
        self._rate_times = list(rate_times)
        self._cf = CashFlow(time_index=0, amount=0.0)

        evolution_times = self._rate_times[:-1]
        self._evolution = EvolutionDescription(self._rate_times, evolution_times)

        n = 0 if not rate_times else len(rate_times) - 1
        if is_exercise_time is None:
            self._is_exercise_time = [True] * n
        else:
            qassert.require(
                len(is_exercise_time) == n,
                f"isExerciseTime ({len(is_exercise_time)}) must have same size as "
                f"rateTimes minus 1 ({n})",
            )
            self._is_exercise_time = list(is_exercise_time)

        self._number_of_exercises = sum(1 for b in self._is_exercise_time if b)
        self._current_index = 0

    def number_of_exercises(self) -> int:
        return self._number_of_exercises

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def possible_cash_flow_times(self) -> list[float]:
        return self._rate_times

    def reset(self) -> None:
        self._current_index = 0

    def next_step(self, current_state: CurveState) -> None:
        self._cf.time_index = self._current_index
        self._current_index += 1

    def is_exercise_time(self) -> list[bool]:
        return self._is_exercise_time

    def value(self, current_state: CurveState) -> CashFlow:
        return self._cf

    def clone(self) -> NothingExerciseValue:
        return copy.deepcopy(self)
