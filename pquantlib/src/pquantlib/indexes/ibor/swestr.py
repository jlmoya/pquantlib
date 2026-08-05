"""Swestr — Swedish krona Short-Term Rate, published by the Riksbank.

# C++ parity: ql/indexes/ibor/swestr.hpp (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.europe import SEKCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.sweden import Sweden


class Swestr(OvernightIndex):
    """SWESTR — SEK overnight risk-free rate."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__("SWESTR", 0, SEKCurrency(), Sweden(), Actual360(), h)
