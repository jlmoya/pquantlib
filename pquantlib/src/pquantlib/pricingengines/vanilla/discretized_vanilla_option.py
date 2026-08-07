"""DiscretizedVanillaOption — lattice helper for plain vanilla options.

# C++ parity: ql/pricingengines/vanilla/discretizedvanillaoption.{hpp,cpp}
# (v1.43).

Ported here because ``DiscretizedDoubleBarrierOption``
(``ql/experimental/barrieroption/discretizeddoublebarrieroption.hpp``)
holds one by value: for the knock-in variants the barrier check needs the
value of the *un-barriered* option at the same lattice node, and the C++
class obtains it by rolling a contained ``DiscretizedVanillaOption`` back
alongside itself.
"""

from __future__ import annotations

import numpy as np

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.methods.lattices.discretized_asset import DiscretizedAsset
from pquantlib.option import OptionArguments
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.time.time_grid import TimeGrid


class DiscretizedVanillaOption(DiscretizedAsset):
    """Vanilla option discretized on a lattice.

    # C++ parity: ``class DiscretizedVanillaOption``
    # (discretizedvanillaoption.hpp:34-51 + .cpp:25-82).
    """

    def __init__(
        self,
        args: OptionArguments,
        process: StochasticProcess,
        grid: TimeGrid | None = None,
    ) -> None:
        super().__init__()
        self._arguments: OptionArguments = args
        exercise = args.exercise
        if exercise is None:
            raise LibraryException("no exercise given")
        # # C++ parity: discretizedvanillaoption.cpp:29-38 — stopping times
        # are the exercise dates mapped through ``process.time``, snapped to
        # the supplied grid when one is given.
        self._stopping_times: list[float] = []
        for d in exercise.dates():
            t = process.time(d)
            if grid is not None and not grid.empty():
                t = grid.closest_time(t)
            self._stopping_times.append(t)

    # --- low-level interface ---------------------------------------------

    def reset(self, size: int) -> None:
        """# C++ parity: ``DiscretizedVanillaOption::reset`` (.cpp:41-44)."""
        self._values = np.zeros(size, dtype=np.float64)
        self.adjust_values()

    def mandatory_times(self) -> list[float]:
        """# C++ parity: ``mandatoryTimes`` (discretizedvanillaoption.hpp:41)."""
        return list(self._stopping_times)

    # --- adjustment ------------------------------------------------------

    def _post_adjust_values_impl(self) -> None:
        """# C++ parity: ``postAdjustValuesImpl`` (.cpp:46-68)."""
        exercise = self._arguments.exercise
        assert exercise is not None
        now = self._time
        et = exercise.type()
        if et == Exercise.Type.American:
            if now <= self._stopping_times[1] and now >= self._stopping_times[0]:
                self._apply_specific_condition()
        elif et == Exercise.Type.European:
            if self.is_on_time(self._stopping_times[0]):
                self._apply_specific_condition()
        elif et == Exercise.Type.Bermudan:
            for stopping_time in self._stopping_times:
                if self.is_on_time(stopping_time):
                    self._apply_specific_condition()
        else:
            raise LibraryException("invalid option type")

    def _apply_specific_condition(self) -> None:
        """# C++ parity: ``applySpecificCondition`` (.cpp:70-78)."""
        method = self._require_method()
        grid = method.grid(self._time)
        payoff = self._arguments.payoff
        assert payoff is not None
        for j in range(self._values.size):
            self._values[j] = max(float(self._values[j]), payoff(float(grid[j])))


__all__ = ["DiscretizedVanillaOption"]
