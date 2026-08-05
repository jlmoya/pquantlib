"""Aonia — Australia Overnight Index Average, fixed by the RBA.

# C++ parity: ql/indexes/ibor/aonia.hpp (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.oceania import AUDCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.australia import Australia


class Aonia(OvernightIndex):
    """AONIA — AUD overnight index average."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__("Aonia", 0, AUDCurrency(), Australia(), Actual365Fixed(), h)
