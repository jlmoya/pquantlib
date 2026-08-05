"""THBFIX — THB onshore fixing derived from the USD/THB swap market.

# C++ parity: ql/indexes/ibor/thbfix.hpp (v1.43). Despite living next to the
LIBOR family upstream, THBFIX derives from ``IborIndex`` directly, so its
fixing calendar is Thailand alone (no London joint calendar).
"""

from __future__ import annotations

from pquantlib.currencies.asia import THBCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.thailand import Thailand
from pquantlib.time.period import Period


class THBFIX(IborIndex):
    """THBFIX — 2 fixing days, ModifiedFollowing, end-of-month on."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(
            "THBFIX", tenor, 2, THBCurrency(), Thailand(),
            BusinessDayConvention.ModifiedFollowing, True, Actual365Fixed(), h,
        )
