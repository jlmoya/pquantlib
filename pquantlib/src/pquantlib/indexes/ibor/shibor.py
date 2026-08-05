"""Shibor — Shanghai Interbank Offered Rate.

# C++ parity: ql/indexes/ibor/shibor.{hpp,cpp} (v1.43).

Two tenor-dependent settings: 0 fixing days for the overnight tenor and 1
otherwise, and ``Following`` for Days/Weeks vs ``ModifiedFollowing`` for
Months/Years. End-of-month is off for every tenor — unlike the LIBOR family,
which turns it on for Months and Years.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.asia import CNYCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.china import China
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def _shibor_convention(p: Period) -> BusinessDayConvention:
    if p.units in (TimeUnit.Days, TimeUnit.Weeks):
        return BusinessDayConvention.Following
    if p.units in (TimeUnit.Months, TimeUnit.Years):
        return BusinessDayConvention.ModifiedFollowing
    qassert.fail("invalid time units")


class Shibor(IborIndex):
    """SHIBOR — CNY interbank offered rate, fixed on the China interbank calendar."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        fixing_days = 0 if tenor == Period(1, TimeUnit.Days) else 1
        super().__init__(
            "Shibor", tenor, fixing_days, CNYCurrency(), China(China.Market.IB),
            _shibor_convention(tenor), False, Actual360(), h,
        )

    def clone(self, forecast_term_structure: YieldTermStructureProtocol | None) -> Shibor:
        """Mirror C++ ``Shibor::clone`` — a fresh Shibor on the same tenor."""
        return Shibor(self._tenor, forecast_term_structure)
