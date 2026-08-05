"""DKKLibor — Danish Krone LIBOR.

# C++ parity: ql/indexes/ibor/dkklibor.hpp (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.europe import DKKCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor.libor import Libor
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.denmark import Denmark
from pquantlib.time.period import Period


class DKKLibor(Libor):
    """DKK ICE LIBOR — 2 fixing days, Denmark financial centre, Actual/360."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__("DKKLibor", tenor, 2, DKKCurrency(), Denmark(), Actual360(), h)
