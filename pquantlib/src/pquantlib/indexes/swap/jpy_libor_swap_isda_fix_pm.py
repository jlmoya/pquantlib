"""JpyLiborSwapIsdaFixPm — JPY LIBOR ISDA-Fix PM swap index.

# C++ parity: ql/indexes/swap/jpyliborswap.{hpp,cpp} (v1.43)

ISDA-Fix PM swap quoted at 3pm Tokyo: semi-annual Actual/Actual (ISDA)
fixed leg against 6M JPY LIBOR at every tenor — no 3M/6M ternary here.

Note the fixing calendar is TARGET, not Japan.
"""

from __future__ import annotations

from pquantlib.currencies.asia import JPYCurrency
from pquantlib.daycounters.actual_actual import ActualActual, Convention
from pquantlib.indexes.ibor.jpy_libor import JPYLibor
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class JpyLiborSwapIsdaFixPm(SwapIndex):
    """JPY LIBOR ISDA-Fix PM (Semi-annual Act/Act ISDA vs 6M JPY LIBOR)."""

    def __init__(
        self,
        tenor: Period,
        forwarding: YieldTermStructureProtocol | None = None,
        discounting: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(
            "JpyLiborSwapIsdaFixPm",
            tenor,
            2,
            JPYCurrency(),
            TARGET(),
            Period(6, TimeUnit.Months),
            BusinessDayConvention.ModifiedFollowing,
            ActualActual(Convention.ISDA),
            JPYLibor(Period(6, TimeUnit.Months), forwarding),
            discounting,
        )
