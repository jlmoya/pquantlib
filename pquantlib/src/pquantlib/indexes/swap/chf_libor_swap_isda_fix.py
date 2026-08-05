"""ChfLiborSwapIsdaFix — CHF LIBOR ISDA-Fix swap index.

# C++ parity: ql/indexes/swap/chfliborswap.{hpp,cpp} (v1.43)

ISDA-Fix swap quoted at 11am Zurich. Annual 30/360 (Bond Basis) fixed leg
against 3M CHF LIBOR for tenors up to and including 1Y, 6M beyond it.

Note the fixing calendar is TARGET, not Switzerland.
"""

from __future__ import annotations

from pquantlib.currencies.europe import CHFCurrency
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.ibor.chf_libor import CHFLibor
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_ONE_YEAR = Period(1, TimeUnit.Years)


def _ibor_for_tenor(
    tenor: Period, h: YieldTermStructureProtocol | None,
) -> CHFLibor:
    """Mirror the C++ ternary: CHFLibor(6M) for tenors > 1Y, else CHFLibor(3M)."""
    if tenor > _ONE_YEAR:
        return CHFLibor(Period(6, TimeUnit.Months), h)
    return CHFLibor(Period(3, TimeUnit.Months), h)



class ChfLiborSwapIsdaFix(SwapIndex):
    """CHF LIBOR ISDA-Fix (Annual 30/360 vs 3M/6M CHF LIBOR)."""

    def __init__(
        self,
        tenor: Period,
        forwarding: YieldTermStructureProtocol | None = None,
        discounting: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(
            "ChfLiborSwapIsdaFix",
            tenor,
            2,
            CHFCurrency(),
            TARGET(),
            _ONE_YEAR,
            BusinessDayConvention.ModifiedFollowing,
            Thirty360(Thirty360Convention.BondBasis),
            _ibor_for_tenor(tenor, forwarding),
            discounting,
        )
