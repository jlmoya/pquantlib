"""Pribor — Prague Interbank Offered Rate, fixed by the Czech National Bank.

# C++ parity: ql/indexes/ibor/pribor.hpp (v1.43).

The number of fixing days is tenor-dependent: 0 for the overnight (1-day)
tenor, 2 otherwise.
"""

from __future__ import annotations

from pquantlib.currencies.europe import CZKCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.czech_republic import CzechRepublic
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class Pribor(IborIndex):
    """PRIBOR — CZK interbank offered rate; ModifiedFollowing, EOM off, Actual/360."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        fixing_days = 0 if tenor == Period(1, TimeUnit.Days) else 2
        super().__init__(
            "PRIBOR", tenor, fixing_days, CZKCurrency(), CzechRepublic(),
            BusinessDayConvention.ModifiedFollowing, False, Actual360(), h,
        )
