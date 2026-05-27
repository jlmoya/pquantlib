"""FlatSmileSection — constant volatility across all strikes at one expiry.

# C++ parity: ql/termstructures/volatility/flatsmilesection.hpp +
#             flatsmilesection.cpp (v1.42.1).

The C++ class returns ``vol_`` regardless of strike. ``min_strike`` is
``QL_MIN_REAL - shift()`` and ``max_strike`` is ``QL_MAX_REAL``; PQuantLib
uses negative / positive infinity (``-math.inf``, ``math.inf``) shifted
appropriately for the same semantics.
"""

from __future__ import annotations

import math

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date


class FlatSmileSection(SmileSection):
    """Constant Black volatility for all strikes at a single expiry."""

    def __init__(
        self,
        *,
        volatility: float,
        exercise_date: Date | None = None,
        exercise_time: float | None = None,
        day_counter: DayCounter | None = None,
        reference_date: Date | None = None,
        atm_level: float | None = None,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        shift: float = 0.0,
    ) -> None:
        super().__init__(
            exercise_date=exercise_date,
            exercise_time=exercise_time,
            day_counter=day_counter,
            reference_date=reference_date,
            volatility_type=volatility_type,
            shift=shift,
        )
        self._vol: float = volatility
        # C++ stores Null<Rate>() if atmLevel is not provided; PQuantLib
        # uses ``math.nan`` as the "not supplied" sentinel and lets
        # callers test ``math.isnan(atm_level())`` if they need to know.
        self._atm_level: float = math.nan if atm_level is None else atm_level

    def min_strike(self) -> float:
        # C++: QL_MIN_REAL - shift(). PQuantLib uses -inf to mean
        # "no lower bound" since -inf - any-finite is still -inf; the
        # shift offset only matters when QL_MIN_REAL is a finite poison
        # value as in the C++ source.
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    def atm_level(self) -> float:
        return self._atm_level

    def _volatility_impl(self, strike: float) -> float:
        _ = strike  # unused — vol is flat
        return self._vol
