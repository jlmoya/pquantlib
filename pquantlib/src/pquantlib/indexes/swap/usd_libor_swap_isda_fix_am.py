"""UsdLiborSwapIsdaFixAm — USDLibor ISDA-Fix AM swap index.

# C++ parity: ql/indexes/swap/usdliborswap.{hpp,cpp} (v1.42.1)

ISDA-Fix AM swap quoted at 11am NYC: Semi-annual 30/360 vs 3M USDLibor.
"""

from __future__ import annotations

from pquantlib.currencies.america import USDCurrency
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.ibor.usd_libor import USDLibor
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.united_states import UnitedStates
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class UsdLiborSwapIsdaFixAm(SwapIndex):
    """USDLibor ISDA-Fix AM (Semi-annual 30/360 vs 3M USDLibor)."""

    def __init__(
        self,
        tenor: Period,
        forwarding: YieldTermStructureProtocol | None = None,
        discounting: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(
            "UsdLiborSwapIsdaFixAm",
            tenor,
            2,
            USDCurrency(),
            UnitedStates(UnitedStates.Market.GovernmentBond),
            Period(6, TimeUnit.Months),
            BusinessDayConvention.ModifiedFollowing,
            Thirty360(Thirty360Convention.BondBasis),
            USDLibor(Period(3, TimeUnit.Months), forwarding),
            discounting,
        )
