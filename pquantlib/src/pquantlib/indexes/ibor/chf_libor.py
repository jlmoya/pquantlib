"""CHFLibor — Swiss Franc LIBOR.

# C++ parity: ql/indexes/ibor/chflibor.hpp (v1.43).
"""

from __future__ import annotations

from pquantlib.currencies.europe import CHFCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor.libor import DailyTenorLibor, Libor
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.switzerland import Switzerland
from pquantlib.time.period import Period


class CHFLibor(Libor):
    """CHF ICE LIBOR — 2 fixing days, Switzerland financial centre, Actual/360."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__("CHFLibor", tenor, 2, CHFCurrency(), Switzerland(), Actual360(), h)


class DailyTenorCHFLibor(DailyTenorLibor):
    """O/N + S/N CHF LIBOR — fixing calendar is joint(UK exchange, Switzerland)."""

    def __init__(
        self, settlement_days: int, h: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(
            "CHFLibor", settlement_days, CHFCurrency(), Switzerland(), Actual360(), h,
        )
