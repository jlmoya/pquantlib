"""AUDLibor — Australian Dollar LIBOR (discontinued as of 2013).

# C++ parity: ql/indexes/ibor/audlibor.hpp (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.oceania import AUDCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor.libor import Libor
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.australia import Australia
from pquantlib.time.period import Period


class AUDLibor(Libor):
    """AUD ICE LIBOR — 2 fixing days, Australia financial centre, Actual/360."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__("AUDLibor", tenor, 2, AUDCurrency(), Australia(), Actual360(), h)
