"""FedFunds — Federal Funds Rate. # C++ parity: ql/indexes/ibor/fedfunds.{hpp,cpp}."""

from __future__ import annotations

from pquantlib.currencies.america import USDCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.united_states import UnitedStates


class FedFunds(OvernightIndex):
    """Fed funds rate (for balances held at the Federal Reserve)."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(
            "FedFunds", 0, USDCurrency(),
            UnitedStates(UnitedStates.Market.FederalReserve),
            Actual360(), h,
        )
