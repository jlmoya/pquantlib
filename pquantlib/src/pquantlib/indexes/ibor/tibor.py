"""Tibor — Tokyo Interbank Offered Rate, fixed by JBA.

# C++ parity: ql/indexes/ibor/tibor.hpp (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.asia import JPYCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.japan import Japan
from pquantlib.time.period import Period


class Tibor(IborIndex):
    """TIBOR — 2 fixing days, ModifiedFollowing, EOM off, Actual/365 (Fixed)."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(
            "Tibor", tenor, 2, JPYCurrency(), Japan(),
            BusinessDayConvention.ModifiedFollowing, False, Actual365Fixed(), h,
        )
