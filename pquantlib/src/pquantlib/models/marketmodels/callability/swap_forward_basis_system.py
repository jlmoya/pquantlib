"""SwapForwardBasisSystem — forward+swap polynomial Longstaff-Schwartz basis.

# C++ parity:
# ql/models/marketmodels/callability/swapforwardbasissystem.{hpp,cpp} (v1.42.1).

A richer LS basis than ``SwapBasisSystem``. Depending on how close the
exercise's rate index is to the terminal rate it returns a 10-, 6- or 3-element
polynomial basis in the forward rate ``x``, the coterminal swap rate ``y`` and
the discount ratio ``z``.

# C++ parity note: ``number_of_functions()`` reports ``{10, ..., 6 or 3}``
# (only the *last* exercise is shortened), but the actual ``values()`` length
# is driven by the per-step rate index — for a middle exercise with
# ``rateIndex == size-3`` ``values()`` returns 6 while ``number_of_functions()``
# still says 10. We reproduce both behaviours verbatim (matching the C++).
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


class SwapForwardBasisSystem(MarketModelBasisSystem):
    """Forward + coterminal-swap polynomial LS basis system.

    # C++ parity: swapforwardbasissystem.hpp SwapForwardBasisSystem.
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
        sizes = [10] * len(self._exercise_times)
        last = self._rate_index[len(self._exercise_times) - 1]
        if last == len(self._rate_times) - 3:
            sizes[-1] = 6
        if last == len(self._rate_times) - 2:
            sizes[-1] = 3
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
        n = len(self._rate_times)
        if rate_index < n - 3:
            x = current_state.forward_rate(rate_index)
            y = current_state.coterminal_swap_rate(rate_index + 1)
            z = current_state.discount_ratio(rate_index, n - 1)
            out = [1.0, x, y, z, x * y, y * z, z * x, x * x, y * y, z * z]
        elif rate_index == n - 3:
            x = current_state.forward_rate(rate_index)
            y = current_state.forward_rate(rate_index + 1)
            out = [1.0, x, y, x * x, x * y, y * y]
        else:
            x = current_state.forward_rate(rate_index)
            out = [1.0, x, x * x]
        results[:] = out

    def clone(self) -> SwapForwardBasisSystem:
        return copy.deepcopy(self)
