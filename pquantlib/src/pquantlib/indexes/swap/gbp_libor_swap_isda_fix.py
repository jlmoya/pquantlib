"""GbpLiborSwapIsdaFix — GBP LIBOR ISDA-Fix swap index.

# C++ parity: ql/indexes/swap/gbpliborswap.{hpp,cpp} (v1.43)

ISDA-Fix swap quoted at 11am London. Unique in this family in that *both* the
fixed-leg tenor and the underlying ibor tenor switch on the swap tenor: at or
below 1Y the fixed leg is annual against 3M GBP LIBOR; above 1Y it is
semi-annual against 6M GBP LIBOR. Fixing days are 0, and the fixing calendar
is the London stock exchange rather than TARGET.
"""

from __future__ import annotations

from pquantlib.currencies.europe import GBPCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor.gbp_libor import GBPLibor
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.united_kingdom import UnitedKingdom
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_ONE_YEAR = Period(1, TimeUnit.Years)
_SIX_MONTHS = Period(6, TimeUnit.Months)
_THREE_MONTHS = Period(3, TimeUnit.Months)


class GbpLiborSwapIsdaFix(SwapIndex):
    """GBP LIBOR ISDA-Fix (Annual vs 3M below 1Y, Semi-annual vs 6M above)."""

    def __init__(
        self,
        tenor: Period,
        forwarding: YieldTermStructureProtocol | None = None,
        discounting: YieldTermStructureProtocol | None = None,
    ) -> None:
        long_tenor = tenor > _ONE_YEAR
        super().__init__(
            "GbpLiborSwapIsdaFix",
            tenor,
            0,
            GBPCurrency(),
            UnitedKingdom(UnitedKingdom.Market.Exchange),
            _SIX_MONTHS if long_tenor else _ONE_YEAR,
            BusinessDayConvention.ModifiedFollowing,
            Actual365Fixed(),
            GBPLibor(_SIX_MONTHS if long_tenor else _THREE_MONTHS, forwarding),
            discounting,
        )
