"""Discretized double-barrier options — lattice helpers.

# C++ parity: ql/experimental/barrieroption/discretizeddoublebarrieroption.{hpp,cpp}
# (v1.43).

Two ``DiscretizedAsset`` subclasses used by ``BinomialDoubleBarrierEngine``:

* :class:`DiscretizedDoubleBarrierOption` — the plain algorithm. Each
  post-adjustment snaps node values to the barrier rules (rebate on the
  knocked-out side, the un-barriered vanilla value on the knocked-in side).
  For every barrier type other than pure knock-out it also rolls a contained
  :class:`~pquantlib.pricingengines.vanilla.discretized_vanilla_option.DiscretizedVanillaOption`
  back in lock-step, which is where the knock-in leg's value comes from.

* :class:`DiscretizedDermanKaniDoubleBarrierOption` — the enhanced algorithm
  of Derman, Kani, Ergener and Bardhan ("Enhanced Numerical Methods for
  Options with Barriers", 1995). It runs the plain algorithm and then
  *interpolates* the two nodes that straddle each barrier, instead of leaving
  them snapped to whatever grid node happens to sit nearby. That removes the
  first-order barrier-quantisation error, which is why the C++ test suite
  accepts it at a tolerance one order of magnitude tighter than the plain
  variant (test-suite/doublebarrieroption.cpp:346-357).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.double_barrier_option import (
    DoubleBarrierOptionArguments,
    DoubleBarrierType,
)
from pquantlib.methods.lattices.discretized_asset import DiscretizedAsset
from pquantlib.pricingengines.vanilla.discretized_vanilla_option import (
    DiscretizedVanillaOption,
)
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.time.time_grid import TimeGrid


class DiscretizedDoubleBarrierOption(DiscretizedAsset):
    """Standard discretized double-barrier option.

    # C++ parity: ``class DiscretizedDoubleBarrierOption``
    # (discretizeddoublebarrieroption.hpp:39-65 + .cpp:25-152).
    """

    def __init__(
        self,
        args: DoubleBarrierOptionArguments,
        process: StochasticProcess,
        grid: TimeGrid | None = None,
    ) -> None:
        super().__init__()
        self._arguments: DoubleBarrierOptionArguments = args
        # # C++ parity: the ``vanilla_(arguments_, process, grid)`` member
        # initialiser runs *before* the ctor body (.cpp:29).
        self._vanilla: DiscretizedVanillaOption = DiscretizedVanillaOption(
            args, process, grid
        )
        exercise = args.exercise
        if exercise is None or not exercise.dates():
            raise LibraryException("specify at least one stopping date")
        self._stopping_times: list[float] = []
        for d in exercise.dates():
            t = process.time(d)
            if grid is not None and not grid.empty():
                t = grid.closest_time(t)
            self._stopping_times.append(t)

    # --- inspectors -------------------------------------------------------

    def vanilla(self) -> npt.NDArray[np.float64]:
        """Node values of the contained un-barriered vanilla option.

        # C++ parity: ``const Array& vanilla() const``
        # (discretizeddoublebarrieroption.hpp:47-49).
        """
        return self._vanilla.values

    def arguments(self) -> DoubleBarrierOptionArguments:
        """# C++ parity: ``arguments()`` (discretizeddoublebarrieroption.hpp:51-53)."""
        return self._arguments

    # --- low-level interface ---------------------------------------------

    def reset(self, size: int) -> None:
        """# C++ parity: ``reset`` (discretizeddoublebarrieroption.cpp:43-47)."""
        method = self._require_method()
        self._vanilla.initialize(method, self._time)
        self._values = np.zeros(size, dtype=np.float64)
        self.adjust_values()

    def mandatory_times(self) -> list[float]:
        """# C++ parity: ``mandatoryTimes`` (discretizeddoublebarrieroption.hpp:55)."""
        return list(self._stopping_times)

    # --- adjustment -------------------------------------------------------

    def _post_adjust_values_impl(self) -> None:
        """# C++ parity: ``postAdjustValuesImpl`` (.cpp:49-55)."""
        if self._arguments.barrier_type != DoubleBarrierType.KnockOut:
            self._vanilla.rollback(self._time)
        method = self._require_method()
        grid = method.grid(self._time)
        self.check_barrier(self._values, grid)

    def check_barrier(  # noqa: PLR0915 — one branch table, one-to-one with C++
        self,
        optvalues: npt.NDArray[np.float64],
        grid: npt.NDArray[np.float64],
    ) -> None:
        """Apply the double-barrier rules to ``optvalues`` in place.

        # C++ parity: ``DiscretizedDoubleBarrierOption::checkBarrier``
        # (discretizeddoublebarrieroption.cpp:57-152).
        """
        arguments = self._arguments
        exercise = arguments.exercise
        assert exercise is not None
        payoff = arguments.payoff
        assert payoff is not None
        barrier_lo = arguments.barrier_lo
        barrier_hi = arguments.barrier_hi
        rebate = arguments.rebate
        assert barrier_lo is not None
        assert barrier_hi is not None
        assert rebate is not None

        end_time = self.is_on_time(self._stopping_times[-1])
        now = self._time
        stopping_time = False
        et = exercise.type()
        if et == Exercise.Type.American:
            # # C++ parity: American reads stoppingTimes_[0] and [1] — a
            # # two-element list is assumed, exactly as in C++ (.cpp:63-67).
            if now <= self._stopping_times[1] and now >= self._stopping_times[0]:
                stopping_time = True
        elif et == Exercise.Type.European:
            if self.is_on_time(self._stopping_times[0]):
                stopping_time = True
        elif et == Exercise.Type.Bermudan:
            for t in self._stopping_times:
                if self.is_on_time(t):
                    stopping_time = True
                    break
        else:
            raise LibraryException("invalid option type")

        barrier_type = arguments.barrier_type
        vanilla = self.vanilla()
        for j in range(optvalues.size):
            gj = float(grid[j])
            if barrier_type == DoubleBarrierType.KnockIn:
                if gj <= barrier_lo or gj >= barrier_hi:
                    # knocked in (dn / up)
                    if stopping_time:
                        optvalues[j] = max(float(vanilla[j]), payoff(gj))
                    else:
                        optvalues[j] = float(vanilla[j])
                elif end_time:
                    optvalues[j] = rebate
            elif barrier_type == DoubleBarrierType.KnockOut:
                if gj <= barrier_lo or gj >= barrier_hi:
                    optvalues[j] = rebate  # knocked out lo / hi
                elif stopping_time:
                    optvalues[j] = max(float(optvalues[j]), payoff(gj))
            elif barrier_type == DoubleBarrierType.KIKO:
                # low barrier is KI, high is KO
                if gj <= barrier_lo:
                    if stopping_time:
                        optvalues[j] = max(float(vanilla[j]), payoff(gj))
                    else:
                        optvalues[j] = float(vanilla[j])
                elif gj >= barrier_hi:
                    optvalues[j] = rebate  # knocked out hi
                elif end_time:
                    optvalues[j] = rebate
            elif barrier_type == DoubleBarrierType.KOKI:
                # low barrier is KO, high is KI
                if gj <= barrier_lo:
                    optvalues[j] = rebate  # knocked out lo
                elif gj >= barrier_hi:
                    if stopping_time:
                        optvalues[j] = max(float(vanilla[j]), payoff(gj))
                    else:
                        optvalues[j] = float(vanilla[j])
                elif end_time:
                    optvalues[j] = rebate
            else:
                raise LibraryException("invalid barrier type")


class DiscretizedDermanKaniDoubleBarrierOption(DiscretizedAsset):
    """Derman-Kani-Ergener-Bardhan enhanced discretized double-barrier option.

    # C++ parity: ``class DiscretizedDermanKaniDoubleBarrierOption``
    # (discretizeddoublebarrieroption.hpp:76-92 + .cpp:156-230).

    Only suitable when the payoff can be approximated linearly across a
    lattice step; the C++ header warns it is unusable for cash-or-nothing
    payoffs.
    """

    def __init__(
        self,
        args: DoubleBarrierOptionArguments,
        process: StochasticProcess,
        grid: TimeGrid | None = None,
    ) -> None:
        super().__init__()
        self._unenhanced: DiscretizedDoubleBarrierOption = (
            DiscretizedDoubleBarrierOption(args, process, grid)
        )

    # --- low-level interface ---------------------------------------------

    def reset(self, size: int) -> None:
        """# C++ parity: ``reset`` (discretizeddoublebarrieroption.cpp:163-167)."""
        method = self._require_method()
        self._unenhanced.initialize(method, self._time)
        self._values = np.zeros(size, dtype=np.float64)
        self.adjust_values()

    def mandatory_times(self) -> list[float]:
        """# C++ parity: ``mandatoryTimes`` (discretizeddoublebarrieroption.hpp:84)."""
        return self._unenhanced.mandatory_times()

    # --- adjustment -------------------------------------------------------

    def _post_adjust_values_impl(self) -> None:
        """# C++ parity: ``postAdjustValuesImpl`` (.cpp:169-175)."""
        self._unenhanced.rollback(self._time)
        method = self._require_method()
        grid = method.grid(self._time)
        self._unenhanced.check_barrier(self._values, grid)  # compute payoffs
        self._adjust_barrier(self._values, grid)

    def _adjust_barrier(
        self,
        optvalues: npt.NDArray[np.float64],
        grid: npt.NDArray[np.float64],
    ) -> None:
        """Interpolate the node values that straddle each barrier.

        # C++ parity: ``adjustBarrier`` (discretizeddoublebarrieroption.cpp:177-230).
        """
        args = self._unenhanced.arguments()
        barrier_lo = args.barrier_lo
        barrier_hi = args.barrier_hi
        rebate = args.rebate
        assert barrier_lo is not None
        assert barrier_hi is not None
        assert rebate is not None
        barrier_type = args.barrier_type

        unenhanced_values = self._unenhanced.values
        n = optvalues.size

        if barrier_type == DoubleBarrierType.KnockIn:
            unenhanced_vanilla = self._unenhanced.vanilla()
            for j in range(n - 1):
                gj = float(grid[j])
                gj1 = float(grid[j + 1])
                if gj <= barrier_lo and gj1 > barrier_lo:
                    # grid[j+1] above barrier_lo, grid[j] under (in),
                    # interpolate optvalues[j+1]
                    ltob = barrier_lo - gj
                    htob = gj1 - barrier_lo
                    htol = gj1 - gj
                    u1 = float(unenhanced_values[j + 1])
                    t1 = float(unenhanced_vanilla[j + 1])
                    optvalues[j + 1] = max(0.0, (ltob * t1 + htob * u1) / htol)
                elif gj < barrier_hi and gj1 >= barrier_hi:
                    # grid[j+1] above barrier_hi (in), grid[j] under,
                    # interpolate optvalues[j]
                    ltob = barrier_hi - gj
                    htob = gj1 - barrier_hi
                    htol = gj1 - gj
                    u = float(unenhanced_values[j])
                    t = float(unenhanced_vanilla[j])
                    optvalues[j] = max(0.0, (ltob * u + htob * t) / htol)
        elif barrier_type == DoubleBarrierType.KnockOut:
            for j in range(n - 1):
                gj = float(grid[j])
                gj1 = float(grid[j + 1])
                if gj <= barrier_lo and gj1 > barrier_lo:
                    a = (barrier_lo - gj) * rebate
                    b = (gj1 - barrier_lo) * float(unenhanced_values[j + 1])
                    c = gj1 - gj
                    optvalues[j + 1] = max(0.0, (a + b) / c)
                elif gj < barrier_hi and gj1 >= barrier_hi:
                    a = (barrier_hi - gj) * float(unenhanced_values[j])
                    b = (gj1 - barrier_hi) * rebate
                    c = gj1 - gj
                    optvalues[j] = max(0.0, (a + b) / c)
        else:
            # # C++ parity: ``QL_FAIL("unsupported barrier type")`` — the
            # # Derman-Kani interpolation is only defined for KnockIn and
            # # KnockOut (.cpp:226-228).
            raise LibraryException("unsupported barrier type")


__all__ = [
    "DiscretizedDermanKaniDoubleBarrierOption",
    "DiscretizedDoubleBarrierOption",
]
