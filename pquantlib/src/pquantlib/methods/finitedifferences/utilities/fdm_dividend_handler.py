"""FdmDividendHandler — discrete-dividend step condition for one equity axis.

# C++ parity: ql/methods/finitedifferences/utilities/fdmdividendhandler.{hpp,cpp}
# @ v1.43 (6b57206e0).

At a dividend time the grid values are re-sampled: the value that used to
sit at spot ``S`` now sits at ``S - d``, so ``a[k]`` is refreshed by
linearly interpolating the pre-drop values at ``max(x[0], x[k] - d)``
(extrapolation allowed, and the ``max`` keeps the lookup off the left edge).

The equity axis is stored in *physical* units — ``x_[i] = exp(location)`` —
so this handler is for log-spot grids.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import final

import numpy as np

from pquantlib.cashflows.dividend import Dividend
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)
from pquantlib.time.date import Date


@final
class FdmDividendHandler(StepCondition):
    """Applies discrete dividends to an FD grid at their payment times.

    # C++ parity: ``class FdmDividendHandler : public StepCondition<Array>``.
    """

    def __init__(
        self,
        schedule: Sequence[Dividend],
        mesher: FdmMesher,
        reference_date: Date,
        day_counter: DayCounter,
        equity_direction: int,
    ) -> None:
        self._mesher: FdmMesher = mesher
        self._equity_direction: int = equity_direction

        self._dividends: list[float] = []
        self._dividend_dates: list[Date] = []
        self._dividend_times: list[float] = []
        for dividend in schedule:
            self._dividends.append(dividend.amount())
            self._dividend_dates.append(dividend.date())
            self._dividend_times.append(
                day_counter.year_fraction(reference_date, dividend.date())
            )

        # C++ parity: ``x_`` is sized ``dim[equityDirection]`` and read out
        # of the *full-layout* ``locations(equityDirection)`` array with a
        # stride of ``spacing[equityDirection]``.
        size = mesher.layout().dim()[equity_direction]
        spacing = mesher.layout().spacing()[equity_direction]
        locations = mesher.locations(equity_direction)
        self._x: Array = np.empty(size, dtype=np.float64)
        for i in range(size):
            self._x[i] = math.exp(float(locations[i * spacing]))

    def dividend_times(self) -> list[float]:
        """# C++ parity: ``dividendTimes()``."""
        return self._dividend_times

    def dividend_dates(self) -> list[Date]:
        """# C++ parity: ``dividendDates()``."""
        return self._dividend_dates

    def dividends(self) -> list[float]:
        """# C++ parity: ``dividends()``."""
        return self._dividends

    def apply_to(self, a: Array, t: float) -> None:
        """Drop the grid by the dividend due exactly at ``t`` (if any).

        # C++ parity: ``FdmDividendHandler::applyTo(Array&, Time)`` — the
        # time lookup is an exact ``std::find`` on the dividend times, so a
        # caller must pass a time taken from :meth:`dividend_times`.
        """
        a_copy = a.copy()

        if t not in self._dividend_times:
            return
        dividend = self._dividends[self._dividend_times.index(t)]

        layout = self._mesher.layout()
        n_x = self._x.shape[0]

        if len(layout.dim()) == 1:
            interp = LinearInterpolation(self._x, a_copy)
            for k in range(n_x):
                a[k] = interp(
                    max(float(self._x[0]), float(self._x[k]) - dividend),
                    allow_extrapolation=True,
                )
        else:
            tmp = np.empty(n_x, dtype=np.float64)
            x_spacing = layout.spacing()[self._equity_direction]

            for i in range(len(layout.dim())):
                if i != self._equity_direction:
                    y_spacing = layout.spacing()[i]
                    for j in range(layout.dim()[i]):
                        for k in range(n_x):
                            tmp[k] = a_copy[j * y_spacing + k * x_spacing]
                        interp = LinearInterpolation(self._x, tmp)
                        for k in range(n_x):
                            a[j * y_spacing + k * x_spacing] = interp(
                                max(float(self._x[0]), float(self._x[k]) - dividend),
                                allow_extrapolation=True,
                            )


__all__ = ["FdmDividendHandler"]
