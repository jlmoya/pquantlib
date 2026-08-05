"""Saron — Swiss Average Rate Overnight.

# C++ parity: ql/indexes/ibor/saron.{hpp,cpp} (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.europe import CHFCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.switzerland import Switzerland


class Saron(OvernightIndex):
    """SARON — CHF overnight secured rate."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__("SARON", 0, CHFCurrency(), Switzerland(), Actual360(), h)
