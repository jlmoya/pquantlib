"""EurLiborSwapIsdaFixA — EUR LIBOR ISDA-Fix A swap index.

# C++ parity: ql/indexes/swap/eurliborswap.{hpp,cpp} (v1.43)

ISDA-Fix A swap quoted at 11am Frankfurt.

Annual 30/360 (Bond Basis) fixed leg against 3M EUR LIBOR for tenors up to
and including 1Y, 6M EUR LIBOR beyond it.
"""

from __future__ import annotations

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.ibor.eur_libor import EURLibor
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_ONE_YEAR = Period(1, TimeUnit.Years)


def _ibor_for_tenor(
    tenor: Period, h: YieldTermStructureProtocol | None,
) -> EURLibor:
    """Mirror the C++ ternary: EURLibor(6M) for tenors > 1Y, else EURLibor(3M)."""
    if tenor > _ONE_YEAR:
        return EURLibor(Period(6, TimeUnit.Months), h)
    return EURLibor(Period(3, TimeUnit.Months), h)



class EurLiborSwapIsdaFixA(SwapIndex):
    """EUR LIBOR ISDA-Fix A (Annual 30/360 vs 3M/6M EUR LIBOR)."""

    def __init__(
        self,
        tenor: Period,
        forwarding: YieldTermStructureProtocol | None = None,
        discounting: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(
            "EurLiborSwapIsdaFixA",
            tenor,
            2,
            EURCurrency(),
            TARGET(),
            _ONE_YEAR,
            BusinessDayConvention.ModifiedFollowing,
            Thirty360(Thirty360Convention.BondBasis),
            _ibor_for_tenor(tenor, forwarding),
            discounting,
        )
