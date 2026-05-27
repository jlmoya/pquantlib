"""ZeroYieldStructure — adapter from zero-yield-impl to discount-impl.

# C++ parity: ql/termstructures/yield/zeroyieldstructure.hpp +
#             ql/termstructures/yield/zeroyieldstructure.cpp (v1.42.1)

Abstract class that lets concrete curves provide ``_zero_yield_impl(t)``
(continuously-compounded zero yield) and derives ``_discount_impl(t)``
automatically as ``exp(-r * t)``. ``InterpolatedZeroCurve``,
``InterpolatedForwardCurve``, ``ForwardSpreadedTermStructure``, and
``ZeroSpreadedTermStructure`` all subclass this.
"""

from __future__ import annotations

import math
from abc import abstractmethod

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class ZeroYieldStructure(YieldTermStructure):
    """Adapter: implement ``_zero_yield_impl``, get discount/forward for free."""

    def __init__(
        self,
        *,
        reference_date: Date | None = None,
        calendar: Calendar | None = None,
        day_counter: DayCounter | None = None,
        jumps: list[Quote] | None = None,
        jump_dates: list[Date] | None = None,
    ) -> None:
        super().__init__(
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
            jumps=jumps,
            jump_dates=jump_dates,
        )

    @abstractmethod
    def _zero_yield_impl(self, t: float) -> float:
        """Continuously-compounded zero yield at time ``t``."""

    def _discount_impl(self, t: float) -> float:
        # C++ parity: zeroyieldstructure.hpp inline ``discountImpl``.
        # Safe guard at t=0 where ``zeroYieldImpl(0)`` might throw.
        if t == 0.0:
            return 1.0
        r = self._zero_yield_impl(t)
        return math.exp(-r * t)
