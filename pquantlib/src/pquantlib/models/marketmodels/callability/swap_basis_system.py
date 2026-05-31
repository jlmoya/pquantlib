"""SwapBasisSystem — coterminal-swap Longstaff-Schwartz basis.

# C++ parity: ql/models/marketmodels/callability/swapbasissystem.{hpp,cpp}
# (v1.42.1).

Per exercise the basis functions are ``{1, forwardRate(rateIndex),
coterminalSwapRate(rateIndex+1)}`` (a 2-vector ``{1, forwardRate}`` when the
exercise's rate index is the second-to-last forward).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from pquantlib.models.marketmodels.callability.market_model_basis_system import (
    MarketModelBasisSystem,
)
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.curve_state import CurveState


class SwapBasisSystem(MarketModelBasisSystem):
    """Coterminal-swap LS basis system.

    # C++ parity: swapbasissystem.hpp SwapBasisSystem.
    """

    def __init__(
        self, rate_times: list[float], exercise_times: list[float]
    ) -> None:
        self._rate_times = list(rate_times)
        self._exercise_times = list(exercise_times)
        self._current_index = 0
        self._evolution = EvolutionDescription(rate_times, exercise_times)

        self._rate_index = [0] * len(exercise_times)
        j = 0
        for i in range(len(exercise_times)):
            while j < len(rate_times) and rate_times[j] < exercise_times[i]:
                j += 1
            self._rate_index[i] = j

    def number_of_exercises(self) -> int:
        return len(self._exercise_times)

    def number_of_functions(self) -> list[int]:
        sizes = [3] * len(self._exercise_times)
        if self._rate_index[len(self._exercise_times) - 1] == len(self._rate_times) - 2:
            sizes[-1] = 2
        return sizes

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def next_step(self, current_state: CurveState) -> None:
        self._current_index += 1

    def reset(self) -> None:
        self._current_index = 0

    def is_exercise_time(self) -> list[bool]:
        return [True] * len(self._exercise_times)

    def values(self, current_state: CurveState, results: list[float]) -> None:
        rate_index = self._rate_index[self._current_index - 1]
        # C++ resize(2) then conditional push_back — replace in place.
        out = [1.0, current_state.forward_rate(rate_index)]
        if rate_index < len(self._rate_times) - 2:
            out.append(current_state.coterminal_swap_rate(rate_index + 1))
        results[:] = out

    def clone(self) -> SwapBasisSystem:
        return copy.deepcopy(self)
