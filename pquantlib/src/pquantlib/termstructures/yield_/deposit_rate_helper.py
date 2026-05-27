"""DepositRateHelper — bootstrap deposit (cash) rate.

# C++ parity: ql/termstructures/yield/ratehelpers.{hpp,cpp} class DepositRateHelper.

Wraps an IborIndex (or constructs a no-fix one from explicit tenor /
fixing days / calendar / convention / EOM / day_counter). The implied
quote is simply ``iborIndex.fixing(fixingDate, forecast_todays_fixing=True)``.

C++ ``RelativeDateRateHelper`` plumbing (auto-re-initialization on
Settings::evaluationDate change) is deferred until Settings becomes
observable in PQuantLib. For now the helper exposes a manual
``initialize_dates(evaluation_date)`` so callers stay in control.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class DepositRateHelper(BootstrapHelper[YieldTermStructureProtocol]):
    """Deposit rate helper. Bootstraps a single discount-curve pillar from a deposit quote."""

    def __init__(
        self,
        rate: Quote | float,
        tenor: Period | None = None,
        fixing_days: int | None = None,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention | None = None,
        end_of_month: bool | None = None,
        day_counter: DayCounter | None = None,
        ibor_index: IborIndex | None = None,
        evaluation_date: Date | None = None,
    ) -> None:
        super().__init__(rate)
        # Two construction modes — mirror the C++ overload set:
        # - explicit (tenor / days / cal / conv / eom / dc), where we
        #   synthesize a "no-fix" IborIndex.
        # - given IborIndex.
        if ibor_index is None:
            qassert.require(
                tenor is not None and fixing_days is not None and calendar is not None
                and convention is not None and end_of_month is not None
                and day_counter is not None,
                "DepositRateHelper: provide either ibor_index OR all of "
                "(tenor, fixing_days, calendar, convention, end_of_month, day_counter)",
            )
            assert tenor is not None
            assert fixing_days is not None
            assert calendar is not None
            assert convention is not None
            assert end_of_month is not None
            assert day_counter is not None
            self._ibor_index = IborIndex(
                "no-fix", tenor, fixing_days, Currency(),
                calendar, convention, end_of_month, day_counter,
            )
        else:
            self._ibor_index = ibor_index

        self._fixing_date: Date | None = None
        if evaluation_date is not None:
            self.initialize_dates(evaluation_date)

    # --- BootstrapHelper interface --------------------------------------------

    def implied_quote(self) -> float:
        """C++ parity: ``DepositRateHelper::impliedQuote``.

        Calls the index's fixing with ``forecast_todays_fixing=True`` to
        bypass historical-fixing lookup.
        """
        qassert.require(self._term_structure is not None, "term structure not set")
        qassert.require(self._fixing_date is not None, "fixing date not initialized")
        assert self._fixing_date is not None
        return self._ibor_index.fixing(self._fixing_date, forecast_todays_fixing=True)

    def set_term_structure(self, ts: YieldTermStructureProtocol) -> None:
        """C++ does linkable-handle swap; PQuantLib just rebinds the index curve."""
        super().set_term_structure(ts)
        self._ibor_index = self._ibor_index.clone(ts)

    # --- dates ----------------------------------------------------------------

    def initialize_dates(self, evaluation_date: Date) -> None:
        """Mirror C++ ``DepositRateHelper::initializeDates`` (relative-date path)."""
        ref = self._ibor_index.fixing_calendar().adjust(
            evaluation_date, BusinessDayConvention.Following,
        )
        earliest = self._ibor_index.value_date(ref)
        self._fixing_date = self._ibor_index.fixing_date(earliest)
        maturity = self._ibor_index.maturity_date(earliest)
        self._earliest_date = earliest
        self._maturity_date = maturity
        self._latest_date = maturity
        self._latest_relevant_date = maturity
        self._pillar_date = maturity
