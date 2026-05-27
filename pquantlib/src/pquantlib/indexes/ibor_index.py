"""IborIndex — base for inter-bank-offered-rate indexes.

# C++ parity: ql/indexes/iborindex.hpp + .cpp (v1.42.1)

Stores a (possibly empty) optional forwarding ``YieldTermStructure`` handle
that satisfies :class:`pquantlib.termstructures.protocols.YieldTermStructureProtocol`.
``forecast_fixing`` computes a forward rate from the curve discount factors.

Subclasses simply pass conventions into the constructor; the L2-C cluster
covers ``Euribor``, ``USDLibor``, ``GBPLibor``, and the overnight family.

C++ ``clone()`` returns a fresh IborIndex bound to a different forwarding
curve — PQuantLib mirrors that with :meth:`IborIndex.clone`.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.interest_rate_index import InterestRateIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class IborIndex(InterestRateIndex):
    """Inter-Bank-Offered-Rate index. May be subclassed for known families."""

    def __init__(
        self,
        family_name: str,
        tenor: Period,
        fixing_days: int,
        currency: Currency,
        fixing_calendar: Calendar,
        convention: BusinessDayConvention,
        end_of_month: bool,
        day_counter: DayCounter,
        forecast_term_structure: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(
            family_name, tenor, fixing_days, currency, fixing_calendar, day_counter
        )
        self._convention: BusinessDayConvention = convention
        self._end_of_month: bool = end_of_month
        self._term_structure: YieldTermStructureProtocol | None = forecast_term_structure

    # --- InterestRateIndex interface ------------------------------------------

    def maturity_date(self, value_date: Date) -> Date:
        """Mirror C++ ``IborIndex::maturityDate`` — advance by tenor with conv/EOM."""
        return self._fixing_calendar.advance(
            value_date, self._tenor.length, self._tenor.units,
            self._convention, self._end_of_month,
        )

    def forecast_fixing(self, fixing_date: Date) -> float:
        """Forecast the index fixing from the term structure."""
        d1 = self.value_date(fixing_date)
        d2 = self.maturity_date(d1)
        t = self._day_counter.year_fraction(d1, d2)
        qassert.require(
            t > 0.0,
            f"cannot calculate forward rate between {d1} and {d2}: "
            f"non positive time ({t}) using {self._day_counter.name()} daycounter",
        )
        return self._forecast_fixing_from_dates(d1, d2, t)

    def _forecast_fixing_from_dates(self, d1: Date, d2: Date, t: float) -> float:
        """C++ parity: ``IborIndex::forecastFixing(d1, d2, t)`` inline (iborindex.hpp).

        Friend-class access (IborCoupon) in C++; in Python we keep it
        underscore-prefixed.
        """
        qassert.require(
            self._term_structure is not None,
            f"null term structure set to this instance of {self.name()}",
        )
        assert self._term_structure is not None
        disc1 = self._term_structure.discount(d1)
        disc2 = self._term_structure.discount(d2)
        return (disc1 / disc2 - 1.0) / t

    # --- inspectors ------------------------------------------------------------

    def business_day_convention(self) -> BusinessDayConvention:
        return self._convention

    def end_of_month(self) -> bool:
        return self._end_of_month

    def forecast_term_structure(self) -> YieldTermStructureProtocol | None:
        return self._term_structure

    # --- mutators --------------------------------------------------------------

    def clone(self, forecast_term_structure: YieldTermStructureProtocol | None) -> IborIndex:
        """Mirror C++ ``IborIndex::clone`` — fresh index bound to a new curve."""
        return IborIndex(
            self._family_name,
            self._tenor,
            self._fixing_days,
            self._currency,
            self._fixing_calendar,
            self._convention,
            self._end_of_month,
            self._day_counter,
            forecast_term_structure,
        )
