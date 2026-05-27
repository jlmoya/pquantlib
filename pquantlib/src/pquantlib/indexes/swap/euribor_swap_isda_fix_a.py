"""EuriborSwapIsdaFixA — Euribor ISDA-Fix A swap index.

# C++ parity: ql/indexes/swap/euriborswap.{hpp,cpp} (v1.42.1)

ISDA-Fix A swap quoted at 11am Frankfurt: Annual 30/360 vs 3M (for tenors
≤ 1Y) or 6M (for tenors > 1Y) Euribor.
"""

from __future__ import annotations

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def _ibor_for_tenor(
    tenor: Period, h: YieldTermStructureProtocol | None,
) -> Euribor:
    """Mirror C++ ternary: Euribor6M for tenors > 1Y, else Euribor3M."""
    if tenor > Period(1, TimeUnit.Years):
        return Euribor(Period(6, TimeUnit.Months), h)
    return Euribor(Period(3, TimeUnit.Months), h)


class EuriborSwapIsdaFixA(SwapIndex):
    """Euribor ISDA-Fix A (Annual 30/360 vs 3M/6M Euribor)."""

    def __init__(
        self,
        tenor: Period,
        forwarding: YieldTermStructureProtocol | None = None,
        discounting: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(
            "EuriborSwapIsdaFixA",
            tenor,
            2,
            EURCurrency(),
            TARGET(),
            Period(1, TimeUnit.Years),
            BusinessDayConvention.ModifiedFollowing,
            Thirty360(Thirty360Convention.BondBasis),
            _ibor_for_tenor(tenor, forwarding),
            discounting,
        )
