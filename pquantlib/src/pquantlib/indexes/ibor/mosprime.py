"""Mosprime — Moscow Prime Offered Rate, fixed by the NFEA.

# C++ parity: ql/indexes/ibor/mosprime.hpp (v1.43).

The number of fixing days is tenor-dependent: 0 for the overnight (1-day)
tenor, 1 otherwise.
"""

from __future__ import annotations

from pquantlib.currencies.europe import RUBCurrency
from pquantlib.daycounters.actual_actual import ActualActual, Convention
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.russia import Russia
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class Mosprime(IborIndex):
    """MOSPRIME — RUB prime offered rate; ModifiedFollowing, EOM off, Actual/Actual (ISDA)."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        fixing_days = 0 if tenor == Period(1, TimeUnit.Days) else 1
        super().__init__(
            "MOSPRIME", tenor, fixing_days, RUBCurrency(), Russia(),
            BusinessDayConvention.ModifiedFollowing, False, ActualActual(Convention.ISDA), h,
        )
