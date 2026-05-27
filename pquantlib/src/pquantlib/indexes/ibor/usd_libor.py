"""USDLibor — USD ICE LIBOR. # C++ parity: ql/indexes/ibor/usdlibor.hpp."""

from __future__ import annotations

from pquantlib.currencies.america import USDCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor.libor import DailyTenorLibor, Libor
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.united_states import UnitedStates
from pquantlib.time.period import Period


class USDLibor(Libor):
    """USD ICE LIBOR — tenor != 1*Days."""

    def __init__(
        self,
        tenor: Period,
        h: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(
            "USDLibor",
            tenor,
            2,
            USDCurrency(),
            UnitedStates(UnitedStates.Market.LiborImpact),
            Actual360(),
            h,
        )


class DailyTenorUSDLibor(DailyTenorLibor):
    """O/N + S/N USD LIBOR."""

    def __init__(
        self,
        fixing_days: int,
        h: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(
            "USDLibor", fixing_days, USDCurrency(),
            UnitedStates(UnitedStates.Market.LiborImpact),
            Actual360(), h,
        )


class USDLiborON(DailyTenorUSDLibor):
    """Overnight USD LIBOR (fixingDays=0)."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(0, h)
