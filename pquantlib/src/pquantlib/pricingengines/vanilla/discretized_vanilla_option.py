"""DiscretizedVanillaOption — vanilla option as a lattice asset.

# C++ parity: ql/pricingengines/vanilla/discretizedvanillaoption.{hpp,cpp}
# (v1.43) — ``class DiscretizedVanillaOption : public DiscretizedAsset``.

The exercise-condition carrier used by the binomial/trinomial tree engines:
``BinomialVanillaEngine<T>`` builds a ``BlackScholesLattice``, wraps the
option's arguments in one of these, and rolls back.

Three responsibilities, all inherited-hook shaped:

``reset(size)``
    zero-fill the value array, then immediately ``adjust_values()`` — which
    is why a European option is worth its payoff at maturity rather than 0.

``mandatory_times()``
    the (grid-snapped) stopping times.

``_post_adjust_values_impl()``
    apply ``max(continuation, payoff(S))`` at the right slices, dispatching
    on exercise type:

    * American — ``stopping[0] <= now <= stopping[1]``: a **range** test, so
      every slice in the window is exercisable.  It indexes ``stopping[1]``,
      so an American exercise must carry two dates (earliest and latest).
    * European — ``is_on_time(stopping[0])``: only the maturity slice.
    * Bermudan — ``is_on_time`` for each stopping time.

The constructor snaps every stopping time onto the supplied grid with
``grid.closest_time(...)``; with no grid the raw ``process.time(date)``
values are kept.  Snapping is observable whenever an exercise date does not
land on a grid point, which the Bermudan cross-validation case exploits.
"""

from __future__ import annotations

import numpy as np

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.methods.lattices.discretized_asset import DiscretizedAsset
from pquantlib.option import OptionArguments
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.time.time_grid import TimeGrid


class DiscretizedVanillaOption(DiscretizedAsset):
    """Vanilla option discretized on a lattice.

    # C++ parity: ``class DiscretizedVanillaOption``
    # (discretizedvanillaoption.hpp:33-49).
    """

    def __init__(
        self,
        args: OptionArguments,
        process: StochasticProcess,
        grid: TimeGrid | None = None,
    ) -> None:
        """Build from the option's engine arguments.

        ``grid`` defaults to C++'s ``TimeGrid()`` — the empty grid, meaning
        "do not snap the stopping times".
        """
        super().__init__()
        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.payoff is not None

        self._arguments: OptionArguments = args
        self._stopping_times: list[float] = []
        for date in args.exercise.dates():
            t = process.time(date)
            if grid is not None and not grid.empty():
                # adjust to the given grid
                t = grid.closest_time(t)
            self._stopping_times.append(t)

    # -- low-level DiscretizedAsset interface ------------------------------

    def reset(self, size: int) -> None:
        """Zero-fill then adjust.

        # C++ parity: ``DiscretizedVanillaOption::reset``.
        """
        self._values = np.zeros(size, dtype=np.float64)
        self.adjust_values()

    def mandatory_times(self) -> list[float]:
        """Grid-snapped stopping times.

        # C++ parity: ``mandatoryTimes() const override
        # { return stoppingTimes_; }``.
        """
        return list(self._stopping_times)

    # -- adjustment hook ---------------------------------------------------

    def _post_adjust_values_impl(self) -> None:
        """Apply the exercise condition at the appropriate slices.

        # C++ parity: ``DiscretizedVanillaOption::postAdjustValuesImpl``.
        """
        now = self.time
        assert self._arguments.exercise is not None
        exercise_type = self._arguments.exercise.type()

        if exercise_type == Exercise.Type.American:
            if now <= self._stopping_times[1] and now >= self._stopping_times[0]:
                self._apply_specific_condition()
        elif exercise_type == Exercise.Type.European:
            if self.is_on_time(self._stopping_times[0]):
                self._apply_specific_condition()
        elif exercise_type == Exercise.Type.Bermudan:
            for stopping_time in self._stopping_times:
                if self.is_on_time(stopping_time):
                    self._apply_specific_condition()
        else:  # pragma: no cover - Exercise.Type has no fourth member
            qassert.fail("invalid option type")

    def _apply_specific_condition(self) -> None:
        """``values[j] = max(values[j], payoff(grid[j]))``.

        # C++ parity: ``DiscretizedVanillaOption::applySpecificCondition``.
        """
        method = self._require_method()
        grid = method.grid(self.time)
        payoff = self._arguments.payoff
        assert payoff is not None
        for j in range(len(self._values)):
            self._values[j] = max(float(self._values[j]), payoff(float(grid[j])))


__all__ = ["DiscretizedVanillaOption"]
