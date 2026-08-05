"""Corra — Canadian Overnight Repo Rate Average.

# C++ parity: ql/indexes/ibor/corra.{hpp,cpp} (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.america import CADCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.canada import Canada


class Corra(OvernightIndex):
    """CORRA — CAD overnight repo rate average."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__("CORRA", 0, CADCurrency(), Canada(), Actual365Fixed(), h)
