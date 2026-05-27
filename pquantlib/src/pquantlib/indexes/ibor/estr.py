"""Estr — Euro Short-Term Rate. # C++ parity: ql/indexes/ibor/estr.{hpp,cpp}."""

from __future__ import annotations

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.calendars.target import TARGET


class Estr(OvernightIndex):
    """ESTR (Euro Short-Term Rate) — ECB fixing."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__("ESTR", 0, EURCurrency(), TARGET(), Actual360(), h)
