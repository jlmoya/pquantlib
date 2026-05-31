"""CallableBondConstantVolatility — flat callable-bond yield vol.

# C++ parity: ql/experimental/callablebonds/callablebondconstantvol.{hpp,cpp}
#             (v1.42.1).

Constant volatility, no time/strike dependence. Wraps a scalar or a
``Quote`` and returns it for every (option_time, bond_length, strike).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.math.constants import QL_MAX_REAL
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.callable_bond_vol_structure import (
    CallableBondVolatilityStructure,
)
from pquantlib.termstructures.volatility.flat_smile_section import FlatSmileSection
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.termstructures.volatility.smile_section import SmileSection
    from pquantlib.time.calendar import Calendar

# C++ uses QL_MIN_REAL (the most-negative finite double) for minStrike.
# Python's QL_MIN_POSITIVE_REAL is the smallest *positive* normal; the
# correct "most negative" sentinel is -QL_MAX_REAL.
_QL_MIN_REAL: float = -QL_MAX_REAL
_HUNDRED_YEARS: Period = Period(100, TimeUnit.Years)


class CallableBondConstantVolatility(CallableBondVolatilityStructure):
    """Constant callable-bond volatility.

    # C++ parity: ``class CallableBondConstantVolatility`` (callablebondconstantvol.hpp:35).

    Four C++ constructor overloads (fixed-date / moving x scalar / quote)
    collapse to a single Python ctor: pass either ``reference_date`` or
    (``settlement_days`` + ``calendar``), and ``volatility`` as a float
    or a ``Quote``.
    """

    def __init__(
        self,
        volatility: float | Quote,
        day_counter: DayCounter,
        *,
        reference_date: Date | None = None,
        settlement_days: int | None = None,
        calendar: Calendar | None = None,
    ) -> None:
        super().__init__(
            reference_date=reference_date,
            settlement_days=settlement_days,
            calendar=calendar,
            day_counter=day_counter,
        )
        quote: Quote = volatility if isinstance(volatility, Quote) else SimpleQuote(volatility)
        self._volatility: Quote = quote
        # Stored separately from the base ``_day_counter`` (which is typed
        # ``DayCounter | None``) so callers get the non-optional accessor.
        self._dc: DayCounter = day_counter
        self._max_bond_tenor: Period = _HUNDRED_YEARS
        self._volatility.register_with(self)

    # ---- TermStructure interface ----

    def day_counter(self) -> DayCounter:
        # C++ parity: callablebondconstantvol.hpp:54.
        return self._dc

    def max_date(self) -> Date:
        # C++ parity: callablebondconstantvol.hpp:55.
        return Date.max_date()

    # ---- CallableBondVolatilityStructure interface ----

    def max_bond_tenor(self) -> Period:
        # C++ parity: callablebondconstantvol.hpp:79-81.
        return self._max_bond_tenor

    def max_bond_length(self) -> float:
        # C++ parity: callablebondconstantvol.hpp:83-85 — QL_MAX_REAL.
        return QL_MAX_REAL

    def min_strike(self) -> float:
        # C++ parity: callablebondconstantvol.hpp:87-89 — QL_MIN_REAL.
        return _QL_MIN_REAL

    def max_strike(self) -> float:
        # C++ parity: callablebondconstantvol.hpp:91-93 — QL_MAX_REAL.
        return QL_MAX_REAL

    # ---- vol implementation ----

    def _volatility_impl(self, option_time: float, bond_length: float, strike: float) -> float:
        # C++ parity: callablebondconstantvol.cpp:65-68.
        _ = (option_time, bond_length, strike)
        return self._volatility.value()

    def _volatility_impl_date(self, option_date: Date, bond_tenor: Period, strike: float) -> float:
        # C++ parity: callablebondconstantvol.cpp:59-63.
        _ = (option_date, bond_tenor, strike)
        return self._volatility.value()

    def _smile_section_impl(self, option_time: float, bond_length: float) -> SmileSection:
        # C++ parity: callablebondconstantvol.cpp:71-79 (FlatSmileSection).
        _ = bond_length
        return FlatSmileSection(
            volatility=self._volatility.value(),
            exercise_time=option_time,
            day_counter=self._dc,
        )

    # Constant volatility has no dependence on bond_length, so override
    # max_bond_length is QL_MAX_REAL; the range check passes for any length.


__all__ = ["CallableBondConstantVolatility"]
