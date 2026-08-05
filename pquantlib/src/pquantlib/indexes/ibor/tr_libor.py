"""TRLibor — Turkish Lira LIBOR (a.k.a. TRLIBOR or TRYLIBOR).

# C++ parity: ql/indexes/ibor/trlibor.hpp (v1.43). Despite the name this is a
plain ``IborIndex``, not a ``Libor``: the fixing calendar is Turkey alone.
"""

from __future__ import annotations

from pquantlib.currencies.europe import TRYCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.turkey import Turkey
from pquantlib.time.period import Period


class TRLibor(IborIndex):
    """TRLIBOR — 0 fixing days, ModifiedFollowing, EOM off, Actual/360."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(
            "TRLibor", tenor, 0, TRYCurrency(), Turkey(),
            BusinessDayConvention.ModifiedFollowing, False, Actual360(), h,
        )
