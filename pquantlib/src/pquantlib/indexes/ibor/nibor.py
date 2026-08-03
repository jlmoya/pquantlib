"""NOK-NIBOR — Norwegian Interbank Offered Rate, new in C++ QuantLib v1.43.

Published via Oslo Boers. Upstream carries a "Check roll convention and EOM"
caveat on this index; the port reproduces its settings verbatim (2 fixing days,
ModifiedFollowing, end_of_month=False, Actual/360).

# C++ parity: ``ql/indexes/ibor/nibor.hpp`` @ v1.43.
"""

from __future__ import annotations

from pquantlib.currencies.europe import NOKCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.norway import Norway
from pquantlib.time.period import Period


class Nibor(IborIndex):
    """NOK-NIBOR index."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(
            "NOK-NIBOR",
            tenor,
            2,
            NOKCurrency(),
            Norway(),
            BusinessDayConvention.ModifiedFollowing,
            False,
            Actual360(),
            h,
        )
