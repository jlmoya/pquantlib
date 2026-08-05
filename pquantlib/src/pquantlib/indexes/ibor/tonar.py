"""Tonar — Tokyo Overnight Average Rate, published by the Bank of Japan.

# C++ parity: ql/indexes/ibor/tonar.hpp (v1.43). The pre-v1.36 spelling
``Tona`` is deprecated upstream and is not ported.
"""

from __future__ import annotations

from pquantlib.currencies.asia import JPYCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.japan import Japan


class Tonar(OvernightIndex):
    """TONAR — JPY overnight average rate."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__("Tonar", 0, JPYCurrency(), Japan(), Actual365Fixed(), h)
