"""Kofr — Korea Overnight Financing Repo rate.

# C++ parity: ql/indexes/ibor/kofr.{hpp,cpp} (v1.43).

Note the fixing calendar is the South-Korean *settlement* calendar, not the
KRX exchange calendar that ``SouthKorea()`` defaults to.
"""

from __future__ import annotations

from pquantlib.currencies.asia import KRWCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.south_korea import SouthKorea


class Kofr(OvernightIndex):
    """KOFR — KRW overnight financing repo rate."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(
            "KOFR", 0, KRWCurrency(),
            SouthKorea(SouthKorea.Market.Settlement), Actual365Fixed(), h,
        )
