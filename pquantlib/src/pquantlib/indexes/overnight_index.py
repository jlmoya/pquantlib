"""OvernightIndex — fixed-1*Day IborIndex with ``Following``/EOM=False.

# C++ parity: ql/indexes/iborindex.hpp lines 88-98 + iborindex.cpp lines 76-94
"""

from __future__ import annotations

from pquantlib.currencies.currency import Currency
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class OvernightIndex(IborIndex):
    """Overnight (1-day) ibor variant. Tenor fixed at ``1*Days``; ``Following``; EOM False."""

    def __init__(
        self,
        family_name: str,
        fixing_days: int,
        currency: Currency,
        fixing_calendar: Calendar,
        day_counter: DayCounter,
        forecast_term_structure: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(
            family_name,
            Period(1, TimeUnit.Days),
            fixing_days,
            currency,
            fixing_calendar,
            BusinessDayConvention.Following,
            False,
            day_counter,
            forecast_term_structure,
        )

    def clone(self, forecast_term_structure: YieldTermStructureProtocol | None) -> OvernightIndex:
        """Mirror C++ ``OvernightIndex::clone`` — fresh instance bound to a curve."""
        return OvernightIndex(
            self._family_name,
            self._fixing_days,
            self._currency,
            self._fixing_calendar,
            self._day_counter,
            forecast_term_structure,
        )
