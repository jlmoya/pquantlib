"""Sofr — Secured Overnight Financing Rate. # C++ parity: ql/indexes/ibor/sofr.{hpp,cpp}."""

from __future__ import annotations

from pquantlib.currencies.america import USDCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.united_states import UnitedStates


class Sofr(OvernightIndex):
    """SOFR (Secured Overnight Financing Rate) — USD overnight, NY Fed fixing."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(
            "SOFR", 0, USDCurrency(),
            UnitedStates(UnitedStates.Market.SOFR), Actual360(), h,
        )
