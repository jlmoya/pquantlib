"""Zibor — Zurich Interbank Offered Rate.

# C++ parity: ql/indexes/ibor/zibor.hpp (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.europe import CHFCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.switzerland import Switzerland
from pquantlib.time.period import Period


class Zibor(IborIndex):
    """ZIBOR — 2 fixing days, ModifiedFollowing, EOM off, Actual/360."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(
            "Zibor", tenor, 2, CHFCurrency(), Switzerland(),
            BusinessDayConvention.ModifiedFollowing, False, Actual360(), h,
        )
