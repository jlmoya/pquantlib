"""Sonia — Sterling Overnight Index Average. # C++ parity: ql/indexes/ibor/sonia.{hpp,cpp}."""

from __future__ import annotations

from pquantlib.currencies.europe import GBPCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.united_kingdom import UnitedKingdom


class Sonia(OvernightIndex):
    """Sterling Overnight Index Average."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(
            "Sonia", 0, GBPCurrency(),
            UnitedKingdom(UnitedKingdom.Market.Exchange),
            Actual365Fixed(), h,
        )
