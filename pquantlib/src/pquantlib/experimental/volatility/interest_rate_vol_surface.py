"""InterestRateVolSurface — interest-rate volatility (smile) surface base.

# C++ parity: ql/experimental/volatility/interestratevolsurface.{hpp,cpp}
# (v1.42.1).

This abstract class extends :class:`BlackVolSurface` with an
:class:`InterestRateIndex` reference. The index drives the
*optionlet-style* date conversion: a tenor ``Period`` is mapped to an
option (fixing) date via the index's fixing calendar, value date, and
fixing-date conventions — overriding the swaption-style conversion in
the base ``VolatilityTermStructure``.

Volatilities are expressed on an annual basis.

Subclasses MUST override the same hooks as :class:`BlackVolSurface`
(``max_date``, ``min_strike``, ``max_strike``, ``_smile_section_impl``).
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.experimental.volatility.black_vol_surface import BlackVolSurface
from pquantlib.indexes.interest_rate_index import InterestRateIndex
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class InterestRateVolSurface(BlackVolSurface):
    """Abstract interest-rate volatility (smile) surface.

    Construction modes 1 (delegated), 2 (fixed reference date) and 3
    (moving via ``settlement_days``) are forwarded, each carrying the
    :class:`InterestRateIndex`.
    """

    def __init__(
        self,
        index: InterestRateIndex,
        *,
        business_day_convention: BusinessDayConvention = BusinessDayConvention.Following,
        reference_date: Date | None = None,
        calendar: Calendar | None = None,
        day_counter: DayCounter | None = None,
        settlement_days: int | None = None,
    ) -> None:
        super().__init__(
            business_day_convention=business_day_convention,
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
            settlement_days=settlement_days,
        )
        self._index: InterestRateIndex = index

    def index(self) -> InterestRateIndex:
        return self._index

    def option_date_from_tenor(self, p: Period) -> Date:
        """Optionlet-style tenor-to-date conversion driven by the index.

        # C++ parity: InterestRateVolSurface::optionDateFromTenor.

        Advances the (calendar-adjusted) reference date to the index
        value date, offsets by the tenor, and reads back the fixing date.
        """
        i = self._index
        ref_date = i.fixing_calendar().adjust(
            self.reference_date(), BusinessDayConvention.Following
        )
        settlement = i.value_date(ref_date)
        # C++: ``Date start = settlement+p;`` — plain Date+Period, no
        # calendar adjustment (the fixing_date read-back below handles it).
        start = settlement + p
        return i.fixing_date(start)


__all__ = ["InterestRateVolSurface"]
