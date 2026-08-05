"""Destr — Denmark Short-Term Rate, published by Danmarks Nationalbank.

# C++ parity: ql/indexes/ibor/destr.hpp (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.europe import DKKCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.denmark import Denmark


class Destr(OvernightIndex):
    """DESTR — DKK overnight risk-free rate."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__("DESTR", 0, DKKCurrency(), Denmark(), Actual360(), h)
