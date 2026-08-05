"""SEKLibor — Swedish Krona LIBOR.

# C++ parity: ql/indexes/ibor/seklibor.hpp (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.europe import SEKCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor.libor import Libor
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.sweden import Sweden
from pquantlib.time.period import Period


class SEKLibor(Libor):
    """SEK ICE LIBOR — 2 fixing days, Sweden financial centre, Actual/360."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__("SEKLibor", tenor, 2, SEKCurrency(), Sweden(), Actual360(), h)
