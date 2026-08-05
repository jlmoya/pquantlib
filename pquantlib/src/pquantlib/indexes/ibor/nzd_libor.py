"""NZDLibor — New Zealand Dollar LIBOR.

# C++ parity: ql/indexes/ibor/nzdlibor.hpp (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.oceania import NZDCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor.libor import Libor
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.new_zealand import NewZealand
from pquantlib.time.period import Period


class NZDLibor(Libor):
    """NZD ICE LIBOR — 2 fixing days, New Zealand financial centre, Actual/360."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__("NZDLibor", tenor, 2, NZDCurrency(), NewZealand(), Actual360(), h)
