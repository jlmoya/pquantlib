"""Nzocr — New Zealand Official Cash Rate, set by the RBNZ.

# C++ parity: ql/indexes/ibor/nzocr.hpp (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.oceania import NZDCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.new_zealand import NewZealand


class Nzocr(OvernightIndex):
    """NZOCR — NZD official cash rate."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__("Nzocr", 0, NZDCurrency(), NewZealand(), Actual365Fixed(), h)
