"""FlatForward — constant-rate yield term structure.

# C++ parity: ql/termstructures/yield/flatforward.hpp +
#             ql/termstructures/yield/flatforward.cpp (v1.42.1)

C++ defines four constructors (date+Quote / date+rate / settlement+Quote
/ settlement+rate). The Python port supports the first two (date-based).
The settlement-days variants depend on ``Settings.evaluation_date``
observability that is deferred per TermStructure carve-out.

Construction style: a single keyword-driven ``__init__`` plus the
classmethod ``from_rate`` for the rate-only convenience constructor
(C++ uses overload sets which Python idiomatically replaces with a
factory pattern).

The class subclasses ``YieldTermStructure`` AND ``LazyObject`` to mirror
the C++ double-inheritance — recomputation of the ``InterestRate``
happens lazily on first ``discount`` call after construction or after
an observer-notified ``update``.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.interest_rate import InterestRate
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency


class FlatForward(YieldTermStructure, LazyObject):
    """Flat (constant) interest-rate curve."""

    def __init__(
        self,
        reference_date: Date,
        forward: Quote,
        day_counter: DayCounter,
        compounding: Compounding = Compounding.Continuous,
        frequency: Frequency = Frequency.Annual,
    ) -> None:
        """Construct from a date and a Quote handle.

        # C++ parity: FlatForward(date, Handle<Quote>, dc, comp, freq).

        Use ``FlatForward.from_rate(...)`` for the convenience constructor
        wrapping a plain float in a SimpleQuote.
        """
        # C++ uses ``Calendar()`` (empty NullCalendar) — we pass None which
        # falls through to TermStructure's none-calendar mode.
        YieldTermStructure.__init__(
            self, reference_date=reference_date, calendar=None, day_counter=day_counter
        )
        LazyObject.__init__(self)
        self._forward: Quote = forward
        self._compounding: Compounding = compounding
        self._frequency: Frequency = frequency
        # Lazy: ``_rate`` is set in ``_perform_calculations``.
        self._rate: InterestRate | None = None
        forward.register_with(self)

    @classmethod
    def from_rate(
        cls,
        reference_date: Date,
        forward_rate: float,
        day_counter: DayCounter,
        compounding: Compounding = Compounding.Continuous,
        frequency: Frequency = Frequency.Annual,
    ) -> FlatForward:
        """Construct from a date and a plain rate (wraps in a SimpleQuote).

        # C++ parity: FlatForward(date, Rate, dc, comp, freq).
        """
        return cls(reference_date, SimpleQuote(forward_rate), day_counter, compounding, frequency)

    # ---- inspectors --------------------------------------------------------

    def compounding(self) -> Compounding:
        return self._compounding

    def compounding_frequency(self) -> Frequency:
        return self._frequency

    def max_date(self) -> Date:
        return Date.max_date()

    # ---- LazyObject + Observer wiring --------------------------------------

    def update(self) -> None:
        # C++ parity: FlatForward inline ``update`` — both LazyObject AND
        # YieldTermStructure parents need to be invalidated. We mirror
        # the call order: LazyObject (cache invalidation) then base.
        LazyObject.update(self)
        YieldTermStructure.update(self)

    def _perform_calculations(self) -> None:
        # C++ parity: ``performCalculations``: build the InterestRate from
        # the current quote value.
        self._rate = InterestRate(
            self._forward.value(), self.day_counter(), self._compounding, self._frequency
        )

    # ---- core discount calculation -----------------------------------------

    def _discount_impl(self, t: float) -> float:
        # C++ parity: inline ``discountImpl(Time)``: call ``calculate()``,
        # then defer to ``rate_.discountFactor(t)``.
        self.calculate()
        assert self._rate is not None
        return self._rate.discount_factor(t)
