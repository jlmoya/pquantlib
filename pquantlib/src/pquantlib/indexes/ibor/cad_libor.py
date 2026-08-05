"""CADLibor — Canadian Dollar LIBOR.

# C++ parity: ql/indexes/ibor/cadlibor.hpp (v1.43).

Note the 0 fixing days — CAD LIBOR settles same-day, unlike most of the LIBOR
family — and Actual/365 (Fixed) rather than Actual/360.
"""

from __future__ import annotations

from pquantlib.currencies.america import CADCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor.libor import DailyTenorLibor, Libor
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.canada import Canada
from pquantlib.time.period import Period


class CADLibor(Libor):
    """CAD ICE LIBOR — 0 fixing days, Canada financial centre, Actual/365 (Fixed)."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__("CADLibor", tenor, 0, CADCurrency(), Canada(), Actual365Fixed(), h)


class CADLiborON(DailyTenorLibor):
    """Overnight CAD LIBOR — fixing calendar is joint(UK exchange, Canada)."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__("CADLibor", 0, CADCurrency(), Canada(), Actual365Fixed(), h)
