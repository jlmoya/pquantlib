"""Cdor — Canadian Dollar Offered Rate, fixed by the IDA.

# C++ parity: ql/indexes/ibor/cdor.hpp (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.america import CADCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.canada import Canada
from pquantlib.time.period import Period


class Cdor(IborIndex):
    """CDOR — CAD offered rate; 0 fixing days, ModifiedFollowing, EOM off."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(
            "CDOR", tenor, 0, CADCurrency(), Canada(),
            BusinessDayConvention.ModifiedFollowing, False, Actual365Fixed(), h,
        )
