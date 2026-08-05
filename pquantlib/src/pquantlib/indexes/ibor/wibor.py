"""Wibor — Warsaw Interbank Offered Rate, fixed by the ACI Polska.

# C++ parity: ql/indexes/ibor/wibor.hpp (v1.43).

The number of fixing days is tenor-dependent: 0 for the overnight (1-day)
tenor, 2 otherwise.
"""

from __future__ import annotations

from pquantlib.currencies.europe import PLNCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.poland import Poland, PolandMarket
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class Wibor(IborIndex):
    """WIBOR — PLN interbank offered rate; ModifiedFollowing, EOM off, Actual/365 (Fixed)."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        fixing_days = 0 if tenor == Period(1, TimeUnit.Days) else 2
        super().__init__(
            "WIBOR", tenor, fixing_days, PLNCurrency(), Poland(PolandMarket.Settlement),
            BusinessDayConvention.ModifiedFollowing, False, Actual365Fixed(), h,
        )
