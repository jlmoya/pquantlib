"""FuturesRateHelper — bootstrap from interest-rate futures price.

# C++ parity: ql/termstructures/yield/ratehelpers.{hpp,cpp} class FuturesRateHelper.

C++ supports three constructor overloads: (price, startDate, lengthInMonths),
(price, startDate, endDate, dayCounter), (price, startDate, iborIndex). We
expose all three via keyword-only arguments.

C++ ``Futures::Type`` is ``IMM``, ``ASX``, or ``Custom``; we honor the IMM
date validity check via the existing ``pquantlib.time.imm`` module.

Convexity adjustment can be either a constant ``float`` or a ``Quote``;
the helper holds an optional Quote wrapper.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.asx import is_asx_date
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.imm import is_imm_date
from pquantlib.time.time_unit import TimeUnit


class FuturesType(IntEnum):
    """C++ parity: ``Futures::Type`` (ql/instruments/futures.hpp)."""

    IMM = 0
    ASX = 1
    Custom = 2


class FuturesRateHelper(BootstrapHelper[YieldTermStructureProtocol]):
    """Bootstrap helper for a futures contract price quote (100 - implied rate)."""

    def __init__(
        self,
        price: Quote | float,
        ibor_start_date: Date,
        length_in_months: int | None = None,
        ibor_end_date: Date | None = None,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention | None = None,
        end_of_month: bool | None = None,
        day_counter: DayCounter | None = None,
        ibor_index: IborIndex | None = None,
        convexity_adjustment: Quote | float = 0.0,
        futures_type: FuturesType = FuturesType.IMM,
    ) -> None:
        super().__init__(price)
        # Validate the start date matches the futures type.
        _check_date(ibor_start_date, futures_type)
        self._convexity_adj: Quote = (
            convexity_adjustment
            if isinstance(convexity_adjustment, Quote)
            else SimpleQuote(float(convexity_adjustment))
        )
        self._earliest_date = ibor_start_date

        if ibor_index is not None:
            cal = ibor_index.fixing_calendar()
            self._maturity_date = cal.advance(
                ibor_start_date, ibor_index.tenor().length, ibor_index.tenor().units,
                ibor_index.business_day_convention(),
            )
            self._year_fraction = ibor_index.day_counter().year_fraction(
                self._earliest_date, self._maturity_date,
            )
        elif length_in_months is not None:
            qassert.require(
                calendar is not None and convention is not None
                and end_of_month is not None and day_counter is not None,
                "FuturesRateHelper: (calendar, convention, end_of_month, day_counter) "
                "required when length_in_months is given",
            )
            assert calendar is not None
            assert convention is not None
            assert end_of_month is not None
            assert day_counter is not None
            self._maturity_date = calendar.advance(
                ibor_start_date, length_in_months, TimeUnit.Months,
                convention, end_of_month,
            )
            self._year_fraction = day_counter.year_fraction(
                self._earliest_date, self._maturity_date,
            )
        elif ibor_end_date is not None:
            qassert.require(
                day_counter is not None,
                "FuturesRateHelper: day_counter required with explicit ibor_end_date",
            )
            assert day_counter is not None
            self._maturity_date = ibor_end_date
            self._year_fraction = day_counter.year_fraction(
                self._earliest_date, self._maturity_date,
            )
        else:
            qassert.fail(
                "FuturesRateHelper: provide one of (length_in_months, ibor_end_date, ibor_index)",
            )
        self._pillar_date = self._maturity_date
        self._latest_date = self._maturity_date
        self._latest_relevant_date = self._maturity_date

    # --- BootstrapHelper interface --------------------------------------------

    def implied_quote(self) -> float:
        qassert.require(self._term_structure is not None, "term structure not set")
        assert self._term_structure is not None
        assert self._earliest_date is not None
        assert self._maturity_date is not None
        forward_rate = (
            self._term_structure.discount(self._earliest_date)
            / self._term_structure.discount(self._maturity_date) - 1.0
        ) / self._year_fraction
        future_rate = forward_rate + self.convexity_adjustment()
        return 100.0 * (1.0 - future_rate)

    def convexity_adjustment(self) -> float:
        return self._convexity_adj.value()


def _check_date(date: Date, futures_type: FuturesType) -> None:
    if futures_type == FuturesType.IMM:
        qassert.require(
            is_imm_date(date, main_cycle=False),
            f"{date} is not a valid IMM date",
        )
    elif futures_type == FuturesType.ASX:
        qassert.require(
            is_asx_date(date, main_cycle=False),
            f"{date} is not a valid ASX date",
        )
