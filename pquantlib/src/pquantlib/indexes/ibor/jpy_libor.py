"""JPYLibor — Japanese Yen LIBOR.

# C++ parity: ql/indexes/ibor/jpylibor.hpp (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.asia import JPYCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor.libor import DailyTenorLibor, Libor
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.japan import Japan
from pquantlib.time.period import Period


class JPYLibor(Libor):
    """JPY ICE LIBOR — 2 fixing days, Japan financial centre, Actual/360."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__("JPYLibor", tenor, 2, JPYCurrency(), Japan(), Actual360(), h)


class DailyTenorJPYLibor(DailyTenorLibor):
    """O/N + S/N JPY LIBOR — fixing calendar is joint(UK exchange, Japan)."""

    def __init__(
        self, settlement_days: int, h: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__("JPYLibor", settlement_days, JPYCurrency(), Japan(), Actual360(), h)
