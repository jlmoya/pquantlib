"""DiscretizedBarrierOption — barrier option as a lattice asset.

# C++ parity: ql/pricingengines/barrier/discretizedbarrieroption.{hpp,cpp}
# (v1.43) — ``class DiscretizedBarrierOption : public DiscretizedAsset`` and
# ``class DiscretizedDermanKaniBarrierOption : public DiscretizedAsset``.

The two carriers :class:`~pquantlib.pricingengines.barrier.binomial_barrier_engine.BinomialBarrierEngine`
rolls back through a :class:`~pquantlib.methods.lattices.bsm_lattice.BlackScholesLattice`.

:class:`DiscretizedBarrierOption`
    Holds a *nested* :class:`~pquantlib.pricingengines.vanilla.discretized_vanilla_option.DiscretizedVanillaOption`
    on the same arguments and grid.  The nested vanilla is what a knock-**in**
    option becomes once it has knocked in, so it is rolled back in lockstep —
    but only for ``DownIn`` / ``UpIn``; for the knock-out types C++ leaves it
    frozen at its maturity values (it is still used by the Derman-Kani
    interpolation, which reads ``vanilla()`` for the knock-in branches only).

    All the behaviour lives in :meth:`DiscretizedBarrierOption.check_barrier`,
    which is public in C++ and public here.  It is *not* symmetric between the
    four barrier types:

    * ``DownIn`` / ``UpIn`` — a knocked node takes ``max(vanilla[j],
      payoff(S))`` at a stopping time and the bare ``vanilla[j]`` otherwise;
      a not-yet-knocked node takes the rebate **only at the terminal slice**
      and is otherwise left holding its rolled-back continuation value.
    * ``DownOut`` / ``UpOut`` — a knocked node is overwritten with the rebate
      at *every* slice, and the live side takes ``max(values[j], payoff(S))``
      at a stopping time.

    The barrier test is ``<=`` for ``Down`` and ``>=`` for ``Up``: a node
    sitting exactly on the barrier counts as knocked.

:class:`DiscretizedDermanKaniBarrierOption`
    Wraps an *unenhanced* :class:`DiscretizedBarrierOption` and adds exactly
    one step — :meth:`~DiscretizedDermanKaniBarrierOption._adjust_barrier`
    linearly interpolates the node **adjacent** to the barrier between the
    unenhanced barrier value and either the nested vanilla value (knock-in) or
    the rebate (knock-out), weighted by where the barrier falls between the two
    grid nodes, floored at 0.  Note the index asymmetry, which is easy to
    "tidy" into a bug: ``Down*`` adjust ``optvalues[j + 1]`` (the node *above*
    the barrier) under the guard ``grid[j] <= barrier < grid[j+1]``, while
    ``Up*`` adjust ``optvalues[j]`` (the node *below*) under the guard
    ``grid[j] < barrier <= grid[j+1]``.

    The adjustment runs **before** ``unenhanced_.checkBarrier(values_, grid)``,
    so ``check_barrier`` overwrites the interpolated node again on the knocked
    side; the visible effect is only on the live side of the barrier.  Unlike
    the plain variant, this one rolls its nested asset back unconditionally,
    for every barrier type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.barrier_option import BarrierOptionArguments, BarrierType
from pquantlib.methods.lattices.discretized_asset import DiscretizedAsset
from pquantlib.pricingengines.vanilla.discretized_vanilla_option import (
    DiscretizedVanillaOption,
)

if TYPE_CHECKING:
    from pquantlib.math.array import Array
    from pquantlib.processes.stochastic_process import StochasticProcess
    from pquantlib.time.time_grid import TimeGrid


class DiscretizedBarrierOption(DiscretizedAsset):
    """Single barrier option discretized on a lattice.

    # C++ parity: ``class DiscretizedBarrierOption``
    # (discretizedbarrieroption.hpp:34-60).
    """

    def __init__(
        self,
        args: BarrierOptionArguments,
        process: StochasticProcess,
        grid: TimeGrid | None = None,
    ) -> None:
        """Build from the option's engine arguments.

        # C++ parity: ``DiscretizedBarrierOption::DiscretizedBarrierOption``
        # (discretizedbarrieroption.cpp:25-41).

        ``grid`` defaults to C++'s ``TimeGrid()`` — the empty grid, meaning
        "do not snap the stopping times".  The nested vanilla is constructed
        first (C++ builds it in the member initialiser list, before the
        ``QL_REQUIRE`` in the body runs).
        """
        super().__init__()
        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.payoff is not None

        self._arguments: BarrierOptionArguments = args
        # C++ parity: ``vanilla_(arguments_, process, grid)`` — member
        # initialiser, so it is built before the check below.
        self._vanilla: DiscretizedVanillaOption = DiscretizedVanillaOption(
            args, process, grid
        )
        qassert.require(
            len(args.exercise.dates()) > 0, "specify at least one stopping date"
        )

        self._stopping_times: list[float] = []
        for date in args.exercise.dates():
            t = process.time(date)
            if grid is not None and not grid.empty():
                # adjust to the given grid
                t = grid.closest_time(t)
            self._stopping_times.append(t)

    # -- inspectors --------------------------------------------------------

    def vanilla(self) -> Array:
        """Node values of the nested vanilla option.

        # C++ parity: ``const Array& vanilla() const``
        # (discretizedbarrieroption.hpp:42-44).
        """
        return self._vanilla.values

    def arguments(self) -> BarrierOptionArguments:
        """The barrier arguments this asset was built from.

        # C++ parity: ``const BarrierOption::arguments& arguments() const``
        # (discretizedbarrieroption.hpp:46-48).
        """
        return self._arguments

    # -- low-level DiscretizedAsset interface ------------------------------

    def reset(self, size: int) -> None:
        """Initialise the nested vanilla, zero-fill, then adjust.

        # C++ parity: ``DiscretizedBarrierOption::reset``
        # (discretizedbarrieroption.cpp:43-47).
        """
        self._vanilla.initialize(self._require_method(), self.time)
        self._values = np.zeros(size, dtype=np.float64)
        self.adjust_values()

    def mandatory_times(self) -> list[float]:
        """Grid-snapped stopping times.

        # C++ parity: ``mandatoryTimes() const override
        # { return stoppingTimes_; }`` (discretizedbarrieroption.hpp:50).
        """
        return list(self._stopping_times)

    # -- adjustment hook ---------------------------------------------------

    def _post_adjust_values_impl(self) -> None:
        """Roll the nested vanilla (knock-in only), then apply the barrier.

        # C++ parity: ``DiscretizedBarrierOption::postAdjustValuesImpl``
        # (discretizedbarrieroption.cpp:49-56).
        """
        if self._arguments.barrier_type in (BarrierType.DownIn, BarrierType.UpIn):
            self._vanilla.rollback(self.time)
        method = self._require_method()
        grid = method.grid(self.time)
        self.check_barrier(self._values, grid)

    def check_barrier(  # noqa: PLR0915 - one-to-one with the C++ body
        self, optvalues: Array, grid: Array
    ) -> None:
        """Apply the barrier rule to ``optvalues`` in place.

        # C++ parity: ``DiscretizedBarrierOption::checkBarrier``
        # (discretizedbarrieroption.cpp:58-131). Public in C++ because
        # ``DiscretizedDermanKaniBarrierOption`` calls it on its own values.
        """
        assert self._arguments.exercise is not None
        payoff = self._arguments.payoff
        assert payoff is not None
        barrier = self._arguments.barrier
        assert barrier is not None
        rebate = self._arguments.rebate
        assert rebate is not None

        now = self.time
        end_time = self.is_on_time(self._stopping_times[-1])
        stopping_time = False
        exercise_type = self._arguments.exercise.type()
        if exercise_type == Exercise.Type.American:
            # A RANGE test, not an ``is_on_time`` one: every slice inside
            # [earliest, latest] is exercisable, so ``stopping_times`` must
            # carry two entries.
            if now <= self._stopping_times[1] and now >= self._stopping_times[0]:
                stopping_time = True
        elif exercise_type == Exercise.Type.European:
            if self.is_on_time(self._stopping_times[0]):
                stopping_time = True
        elif exercise_type == Exercise.Type.Bermudan:
            for t in self._stopping_times:
                if self.is_on_time(t):
                    stopping_time = True
                    break
        else:  # pragma: no cover - Exercise.Type has no fourth member
            qassert.fail("invalid option type")

        barrier_type = self._arguments.barrier_type
        vanilla_values = self._vanilla.values
        for j in range(len(optvalues)):
            s = float(grid[j])
            if barrier_type == BarrierType.DownIn:
                if s <= barrier:
                    # knocked in
                    if stopping_time:
                        optvalues[j] = max(float(vanilla_values[j]), payoff(s))
                    else:
                        optvalues[j] = float(vanilla_values[j])
                elif end_time:
                    optvalues[j] = rebate
            elif barrier_type == BarrierType.DownOut:
                if s <= barrier:
                    optvalues[j] = rebate  # knocked out
                elif stopping_time:
                    optvalues[j] = max(float(optvalues[j]), payoff(s))
            elif barrier_type == BarrierType.UpIn:
                if s >= barrier:
                    # knocked in
                    if stopping_time:
                        optvalues[j] = max(float(vanilla_values[j]), payoff(s))
                    else:
                        optvalues[j] = float(vanilla_values[j])
                elif end_time:
                    optvalues[j] = rebate
            elif barrier_type == BarrierType.UpOut:
                if s >= barrier:
                    optvalues[j] = rebate  # knocked out
                elif stopping_time:
                    optvalues[j] = max(float(optvalues[j]), payoff(s))
            else:  # pragma: no cover - BarrierType has no fifth member
                qassert.fail("invalid barrier type")


class DiscretizedDermanKaniBarrierOption(DiscretizedAsset):
    """Derman-Kani barrier-node-interpolated variant.

    # C++ parity: ``class DiscretizedDermanKaniBarrierOption``
    # (discretizedbarrieroption.hpp:62-78).
    """

    def __init__(
        self,
        args: BarrierOptionArguments,
        process: StochasticProcess,
        grid: TimeGrid | None = None,
    ) -> None:
        """# C++ parity:
        # ``DiscretizedDermanKaniBarrierOption::DiscretizedDermanKaniBarrierOption``
        # (discretizedbarrieroption.cpp:135-140)."""
        super().__init__()
        self._unenhanced: DiscretizedBarrierOption = DiscretizedBarrierOption(
            args, process, grid
        )

    # -- low-level DiscretizedAsset interface ------------------------------

    def reset(self, size: int) -> None:
        """# C++ parity: ``DiscretizedDermanKaniBarrierOption::reset``
        # (discretizedbarrieroption.cpp:142-146)."""
        self._unenhanced.initialize(self._require_method(), self.time)
        self._values = np.zeros(size, dtype=np.float64)
        self.adjust_values()

    def mandatory_times(self) -> list[float]:
        """# C++ parity: ``mandatoryTimes() const override
        # { return unenhanced_.mandatoryTimes(); }``
        # (discretizedbarrieroption.hpp:70)."""
        return self._unenhanced.mandatory_times()

    # -- adjustment hook ---------------------------------------------------

    def _post_adjust_values_impl(self) -> None:
        """Roll the unenhanced option, interpolate at the barrier, then apply it.

        # C++ parity:
        # ``DiscretizedDermanKaniBarrierOption::postAdjustValuesImpl``
        # (discretizedbarrieroption.cpp:148-154). Note the unenhanced asset is
        # rolled back for EVERY barrier type here, unlike the plain variant
        # which only rolls its nested vanilla for the knock-in types.
        """
        self._unenhanced.rollback(self.time)
        method = self._require_method()
        grid = method.grid(self.time)
        self._adjust_barrier(self._values, grid)
        self._unenhanced.check_barrier(self._values, grid)  # compute payoffs

    def _adjust_barrier(self, optvalues: Array, grid: Array) -> None:
        """Interpolate the node adjacent to the barrier, in place.

        # C++ parity: ``DiscretizedDermanKaniBarrierOption::adjustBarrier``
        # (discretizedbarrieroption.cpp:156-213).

        Note the C++ ``switch`` has no ``default``: an out-of-range barrier
        type silently does nothing rather than failing, which is reproduced.
        """
        args = self._unenhanced.arguments()
        barrier = args.barrier
        assert barrier is not None
        rebate = args.rebate
        assert rebate is not None
        barrier_type = args.barrier_type
        unenhanced = self._unenhanced.values
        vanilla = self._unenhanced.vanilla()

        n = len(optvalues)
        if barrier_type == BarrierType.DownIn:
            for j in range(n - 1):
                if grid[j] <= barrier and grid[j + 1] > barrier:
                    # grid[j+1] above barrier, grid[j] under (in),
                    # interpolate optvalues[j+1]
                    ltob = barrier - float(grid[j])
                    htob = float(grid[j + 1]) - barrier
                    htol = float(grid[j + 1]) - float(grid[j])
                    u1 = float(unenhanced[j + 1])
                    t1 = float(vanilla[j + 1])
                    optvalues[j + 1] = max(0.0, (ltob * t1 + htob * u1) / htol)
        elif barrier_type == BarrierType.DownOut:
            for j in range(n - 1):
                if grid[j] <= barrier and grid[j + 1] > barrier:
                    # grid[j+1] above barrier, grid[j] under (out),
                    # interpolate optvalues[j+1]
                    a = (barrier - float(grid[j])) * rebate
                    b = (float(grid[j + 1]) - barrier) * float(unenhanced[j + 1])
                    c = float(grid[j + 1]) - float(grid[j])
                    optvalues[j + 1] = max(0.0, (a + b) / c)
        elif barrier_type == BarrierType.UpIn:
            for j in range(n - 1):
                if grid[j] < barrier and grid[j + 1] >= barrier:
                    # grid[j+1] above barrier (in), grid[j] under,
                    # interpolate optvalues[j]
                    ltob = barrier - float(grid[j])
                    htob = float(grid[j + 1]) - barrier
                    htol = float(grid[j + 1]) - float(grid[j])
                    u = float(unenhanced[j])
                    t = float(vanilla[j])
                    optvalues[j] = max(0.0, (ltob * u + htob * t) / htol)  # derman std
        elif barrier_type == BarrierType.UpOut:
            for j in range(n - 1):
                if grid[j] < barrier and grid[j + 1] >= barrier:
                    # grid[j+1] above barrier (out), grid[j] under,
                    # interpolate optvalues[j]
                    a = (barrier - float(grid[j])) * float(unenhanced[j])
                    b = (float(grid[j + 1]) - barrier) * rebate
                    c = float(grid[j + 1]) - float(grid[j])
                    optvalues[j] = max(0.0, (a + b) / c)


__all__ = ["DiscretizedBarrierOption", "DiscretizedDermanKaniBarrierOption"]
