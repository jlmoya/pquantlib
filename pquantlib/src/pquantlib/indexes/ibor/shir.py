"""SHIR — Shekel Overnight Interest Rate, new in C++ QuantLib v1.43.

Published by the Bank of Israel, replacing the Telbor rate in interest-rate
derivative transactions; SHIR is the overnight rate for that day (same-day
fixing). See https://www.boi.org.il/en/economic-roles/financial-markets/shir/

# C++ parity: ``ql/indexes/ibor/shir.{hpp,cpp}`` @ v1.43.
"""

from __future__ import annotations

from pquantlib.currencies.asia import ILSCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.israel import Israel, IsraelMarket


class Shir(OvernightIndex):
    """Shekel Overnight Interest Rate."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(
            "Shir",
            0,
            ILSCurrency(),
            Israel(IsraelMarket.SHIR),
            Actual365Fixed(),
            h,
        )
