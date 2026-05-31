"""BermudanSwaptionExerciseValue — exercise value of a Bermudan swaption.

# C++ parity:
# ql/models/marketmodels/callability/bermudanswaptionexercisevalue.{hpp,cpp}
# (v1.42.1).

At each exercise opportunity ``k`` the value received is the (floored) value of
entering the coterminal swap starting at ``k``:

    max( coterminalSwapAnnuity(k, k) * payoff_k(coterminalSwapRate(k)), 0 ).
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
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
    from pquantlib.payoffs import Payoff


class BermudanSwaptionExerciseValue(MarketModelExerciseValue):
    """Exercise value of a Bermudan swaption over coterminal swaps.

    # C++ parity: bermudanswaptionexercisevalue.hpp BermudanSwaptionExerciseValue.
    """

    def __init__(self, rate_times: list[float], payoffs: Sequence[Payoff]) -> None:
        check_increasing_times(rate_times)
        self._number_of_exercises = 0 if not rate_times else len(rate_times) - 1
        qassert.require(
            self._number_of_exercises > 0,
            "Rate times must contain at least two values",
        )
        self._rate_times = list(rate_times)
        self._payoffs = list(payoffs)
        self._cf = CashFlow(time_index=0, amount=0.0)

        evolve_times = self._rate_times[:-1]
        self._evolution = EvolutionDescription(self._rate_times, evolve_times)
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
        p = self._payoffs[self._current_index]
        value = current_state.coterminal_swap_annuity(
            self._current_index, self._current_index
        ) * p(current_state.coterminal_swap_rate(self._current_index))
        value = max(value, 0.0)
        self._cf.time_index = self._current_index
        self._cf.amount = value
        self._current_index += 1

    def is_exercise_time(self) -> list[bool]:
        return [True] * self._number_of_exercises

    def value(self, current_state: CurveState) -> CashFlow:
        return self._cf

    def clone(self) -> BermudanSwaptionExerciseValue:
        return copy.deepcopy(self)
