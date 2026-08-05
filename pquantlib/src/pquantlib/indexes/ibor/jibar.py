"""Jibar — Johannesburg Interbank Agreed Rate, fixed by SAFEX.

# C++ parity: ql/indexes/ibor/jibar.hpp (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.africa import ZARCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.south_africa import SouthAfrica
from pquantlib.time.period import Period


class Jibar(IborIndex):
    """JIBAR — ZAR interbank agreed rate; 0 fixing days, ModifiedFollowing, EOM off."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(
            "Jibar", tenor, 0, ZARCurrency(), SouthAfrica(),
            BusinessDayConvention.ModifiedFollowing, False, Actual365Fixed(), h,
        )
