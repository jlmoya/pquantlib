"""ForwardSwapQuote — par rate of a forward-starting swap.

# C++ parity: ql/quotes/forwardswapquote.{hpp,cpp} (v1.43)

The quote snaps three dates off the evaluation date —

* ``valueDate``  = fixing calendar advanced by the index's fixing days,
* ``startDate``  = ``valueDate`` advanced by ``fwdStart``,
* ``fixingDate`` = the index's fixing date for ``startDate`` —

builds the swap index's underlying swap at that fixing date, and solves for the
fixed rate that zeroes the swap once the floating leg is shifted by ``spread``.
It registers with the evaluation date, so rolling the date re-snaps all three
dates and rebuilds the swap.

# C++ parity divergence (Handle vs object): C++ takes a ``Handle<Quote>``
spread. This port threads the ``Quote`` directly (see
``cashflows/cms_coupon_pricer.py``); ``None`` models C++'s *empty* handle,
which both ``performCalculations`` (spread of zero) and ``isValid`` (skip the
spread check) explicitly test for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.quote import Quote
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.indexes.swap_index import SwapIndex
    from pquantlib.instruments.fixed_vs_floating_swap import FixedVsFloatingSwap
    from pquantlib.time.date import Date
    from pquantlib.time.period import Period

# C++ ``static const Spread basisPoint = 1.0e-4;`` in performCalculations.
_BASIS_POINT: float = 1.0e-4


class ForwardSwapQuote(Quote, LazyObject):
    """Quote for a forward starting swap."""

    def __init__(
        self,
        swap_index: SwapIndex,
        spread: Quote | None,
        fwd_start: Period,
    ) -> None:
        super().__init__()
        self._swap_index: SwapIndex = swap_index
        self._spread: Quote | None = spread
        self._fwd_start: Period = fwd_start
        self._result: float = 0.0
        swap_index.register_with(self)
        if spread is not None:
            spread.register_with(self)
        settings = ObservableSettings()
        settings.register_with(self)
        self._evaluation_date: Date = settings.evaluation_date_or_today()
        self._value_date: Date
        self._start_date: Date
        self._fixing_date: Date
        self._swap: FixedVsFloatingSwap
        self._initialize_dates()

    def _initialize_dates(self) -> None:
        calendar = self._swap_index.fixing_calendar()
        self._value_date = calendar.advance(
            self._evaluation_date,
            self._swap_index.fixing_days(),
            TimeUnit.Days,
            BusinessDayConvention.Following,
        )
        self._start_date = calendar.advance_period(
            self._value_date, self._fwd_start, BusinessDayConvention.Following
        )
        self._fixing_date = self._swap_index.fixing_date(self._start_date)
        self._swap = self._swap_index.underlying_swap(self._fixing_date)

    # --- Quote interface --------------------------------------------------

    def value(self) -> float:
        self.calculate()
        return self._result

    def is_valid(self) -> bool:
        swap_index_is_valid = True
        try:
            self._swap.recalculate()
        except Exception:
            # C++ ``catch (...)``: any failure at all means "not valid".
            swap_index_is_valid = False
        spread_is_valid = True if self._spread is None else self._spread.is_valid()
        return swap_index_is_valid and spread_is_valid

    # --- Observer interface -----------------------------------------------

    def update(self) -> None:
        current = ObservableSettings().evaluation_date_or_today()
        if self._evaluation_date != current:
            self._evaluation_date = current
            self._initialize_dates()
        super().update()

    # --- inspectors -------------------------------------------------------

    def value_date(self) -> Date:
        self.calculate()
        return self._value_date

    def start_date(self) -> Date:
        self.calculate()
        return self._start_date

    def fixing_date(self) -> Date:
        self.calculate()
        return self._fixing_date

    # --- LazyObject interface ---------------------------------------------

    def _perform_calculations(self) -> None:
        # C++ comment, verbatim: "we didn't register as observers - force
        # calculation".
        self._swap.recalculate()
        # C++ comment, verbatim: "weak implementation... to be improved".
        floating_leg_npv = self._swap.floating_leg_npv()
        spread = 0.0 if self._spread is None else self._spread.value()
        spread_npv = self._swap.floating_leg_bps() / _BASIS_POINT * spread
        tot_npv = -(floating_leg_npv + spread_npv)
        self._result = tot_npv / (self._swap.fixed_leg_bps() / _BASIS_POINT)
