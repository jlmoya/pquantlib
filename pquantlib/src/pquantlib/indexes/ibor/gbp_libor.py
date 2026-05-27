"""GBPLibor — Sterling ICE LIBOR. # C++ parity: ql/indexes/ibor/gbplibor.hpp."""

from __future__ import annotations

from pquantlib.currencies.europe import GBPCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor.libor import DailyTenorLibor, Libor
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.united_kingdom import UnitedKingdom
from pquantlib.time.period import Period


class GBPLibor(Libor):
    """GBP ICE LIBOR — Sterling, fixingDays=0, Actual/365 (Fixed)."""

    def __init__(
        self,
        tenor: Period,
        h: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(
            "GBPLibor",
            tenor,
            0,
            GBPCurrency(),
            UnitedKingdom(UnitedKingdom.Market.Exchange),
            Actual365Fixed(),
            h,
        )


class DailyTenorGBPLibor(DailyTenorLibor):
    """O/N + S/N GBP LIBOR."""

    def __init__(
        self,
        fixing_days: int,
        h: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(
            "GBPLibor", fixing_days, GBPCurrency(),
            UnitedKingdom(UnitedKingdom.Market.Exchange),
            Actual365Fixed(), h,
        )


class GBPLiborON(DailyTenorGBPLibor):
    """Overnight GBP LIBOR (fixingDays=0)."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(0, h)
