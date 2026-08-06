"""FdmArithmeticAverageCondition — fixing-date update for arithmetic Asians.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmarithmeticaveragecondition.{hpp,cpp}
# (v1.43).

The mesh is 2-D: one direction carries the (log) equity price, the other the
(log) running arithmetic average. Both are stored in log units, so the
condition exponentiates them once in the constructor into ``x_`` (equity, in
physical units) and ``a_`` (average, in physical units).

At a fixing date the running average jumps from ``A`` to

    A' = (i_T - n) / i_T * A + n / i_T * S

where ``i_T`` is the *total* number of fixings observed up to and including
this date (past fixings plus the position of this date in the schedule) and
``n`` is the number of fixings that land on this very date. Backward
induction therefore has to re-read the solution at the *pre-jump* average
that maps to each post-jump grid average, which is an interpolation along
the average direction — C++ uses ``MonotonicCubicNaturalSpline`` with
``allowExtrapolation = true``.

Extrapolation is not incidental
-------------------------------
``A'`` is a convex combination of a grid average and a grid *equity* value.
Whenever the equity grid extends beyond the average grid — which it normally
does — ``A'`` falls outside ``[a_[0], a_[-1]]`` at the extreme equity nodes,
so the ``true`` in ``interp(..., true)`` is doing real work. C++ extrapolates
by continuing the edge cubic (``CubicInterpolation::Impl::value`` clamps only
the *bracket index*, then evaluates the polynomial), which can and does
produce values far outside the data range. PQuantLib's ``CubicInterpolation``
reproduces that; a clamping interpolator such as ``numpy.interp`` would not.

Constructor argument that C++ ignores
-------------------------------------
The second constructor argument is unnamed in C++ and never stored. Its only
call site (``FdBlackScholesAsianEngine``) passes
``arguments_.runningAccumulator``, so the port keeps it under that name for
signature parity and documents that it is unused.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.cubic_interpolation import MonotonicCubicNaturalSpline
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)


@final
class FdmArithmeticAverageCondition(StepCondition):
    """Arithmetic-average fixing update on a 2-D (equity, average) mesh.

    # C++ parity: ``class FdmArithmeticAverageCondition : public StepCondition<Array>``.
    """

    __slots__ = (
        "_a",
        "_average_times",
        "_equity_direction",
        "_mesher",
        "_past_fixings",
        "_x",
    )

    def __init__(
        self,
        average_times: Sequence[float],
        running_accumulator: float,
        past_fixings: int,
        mesher: FdmMesher,
        equity_direction: int,
    ) -> None:
        # C++ validates in the constructor *body*, i.e. after the member
        # initialiser list has already indexed dim()[...]. Python validates
        # first: on a bad direction or rank the C++ order is undefined
        # behaviour, so there is nothing to be faithful to.
        layout = mesher.layout()
        qassert.require(len(layout.dim()) == 2, "2D allowed only")
        qassert.require(
            equity_direction in {0, 1},
            "equityDirection has to be 0 or 1",
        )

        del running_accumulator  # C++ parity: the Real argument is unnamed and unused.

        self._average_times: list[float] = list(average_times)
        self._past_fixings: int = past_fixings
        self._mesher: FdmMesher = mesher
        self._equity_direction: int = equity_direction

        average_direction = 1 if equity_direction == 0 else 0

        # x_: grid-equity values in physical units.
        x_spacing = layout.spacing()[equity_direction]
        tmp = mesher.locations(equity_direction)
        self._x: Array = np.array(
            [math.exp(float(tmp[i * x_spacing])) for i in range(layout.dim()[equity_direction])],
            dtype=np.float64,
        )

        # a_: grid-average values in physical units.
        a_spacing = layout.spacing()[average_direction]
        tmp = mesher.locations(average_direction)
        self._a: Array = np.array(
            [math.exp(float(tmp[i * a_spacing])) for i in range(layout.dim()[average_direction])],
            dtype=np.float64,
        )

    def apply_to(self, a: Array, t: float) -> None:
        """Apply the fixing-date average update at ``t``.

        # C++ parity: ``FdmArithmeticAverageCondition::applyTo``.
        """
        layout = self._mesher.layout()
        # C++ asserts unconditionally, before testing the time.
        qassert.require(
            layout.size() == a.size,
            f"inconsistent array dimensions: a has {a.size}, layout has {layout.size()}",
        )

        n_times = self._average_times.count(t)
        if n_times == 0:
            return

        a_copy = a.copy()
        # C++ `iter - averageTimes_.begin()` is the index of the FIRST match,
        # even when the date is repeated.
        i_t = self._average_times.index(t) + 1 + self._past_fixings

        average_direction = 1 if self._equity_direction == 0 else 0
        x_spacing = layout.spacing()[self._equity_direction]
        a_spacing = layout.spacing()[average_direction]

        n_x = int(self._x.shape[0])
        n_a = int(self._a.shape[0])

        # Same operand grouping as C++:
        #   (iT-nTimes)/(double)(iT) * a_[j] + nTimes/(double)(iT) * x_[i]
        w_average = (i_t - n_times) / float(i_t)
        w_equity = n_times / float(i_t)

        for i in range(n_x):
            # Allocated per column: PQuantLib's Interpolation base does not
            # necessarily copy an already-contiguous float64 input, so a
            # reused buffer would alias the live interpolant's y data.
            tmp: Array = np.empty(n_a, dtype=np.float64)
            for j in range(n_a):
                tmp[j] = a_copy[i * x_spacing + j * a_spacing]

            interp = MonotonicCubicNaturalSpline(self._a, tmp)

            x_i = float(self._x[i])
            for j in range(n_a):
                index = i * x_spacing + j * a_spacing
                a[index] = interp(
                    w_average * float(self._a[j]) + w_equity * x_i,
                    allow_extrapolation=True,
                )


__all__ = ["FdmArithmeticAverageCondition"]
