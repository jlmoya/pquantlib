"""ZARONIA — South African Rand Overnight Index Average, new in C++ QuantLib v1.43.

Fixed by the South African Reserve Bank.

# C++ parity: ``ql/indexes/ibor/zaronia.{hpp,cpp}`` @ v1.43.
"""

from __future__ import annotations

from pquantlib.currencies.africa import ZARCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.south_africa import SouthAfrica


class Zaronia(OvernightIndex):
    """South African Rand Overnight Index Average."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(
            "ZARONIA",
            0,
            ZARCurrency(),
            SouthAfrica(),
            Actual365Fixed(),
            h,
        )
