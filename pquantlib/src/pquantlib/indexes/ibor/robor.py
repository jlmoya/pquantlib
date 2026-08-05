"""Robor — Romanian Interbank Offered Rate, fixed by the National Bank of Romania.

# C++ parity: ql/indexes/ibor/robor.hpp (v1.43).

The number of fixing days is tenor-dependent: 0 for the overnight (1-day)
tenor, 2 otherwise.
"""

from __future__ import annotations

from pquantlib.currencies.europe import RONCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.romania import Romania
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class Robor(IborIndex):
    """ROBOR — RON interbank offered rate; ModifiedFollowing, EOM off, Actual/360."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        fixing_days = 0 if tenor == Period(1, TimeUnit.Days) else 2
        super().__init__(
            "ROBOR", tenor, fixing_days, RONCurrency(), Romania(),
            BusinessDayConvention.ModifiedFollowing, False, Actual360(), h,
        )
