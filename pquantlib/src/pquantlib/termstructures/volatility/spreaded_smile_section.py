"""SpreadedSmileSection — base smile plus a Quote-driven additive vol spread.

# C++ parity: ql/termstructures/volatility/spreadedsmilesection.{hpp,cpp}
# (v1.42.1).

The C++ class wraps an existing ``SmileSection`` and a ``Handle<Quote>``;
``volatilityImpl(K)`` returns ``base.volatility(K) + spread.value()``.
Every other accessor (min/max strike, atm_level, exercise_time,
day_counter, reference_date, volatility_type, shift) delegates to the
base section. The class registers as an observer of both the base and
the spread Quote so it propagates updates.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date


class SpreadedSmileSection(SmileSection):
    """Smile section = base + additive vol spread (Quote-driven).

    Args:
        base: underlying ``SmileSection``.
        vol_spread: ``Quote`` returning the additive vol spread.
    """

    def __init__(
        self,
        *,
        base: SmileSection,
        vol_spread: Quote,
    ) -> None:
        # We don't call SmileSection.__init__ in the standard way —
        # the C++ parent class is shadowed entirely by ``underlyingSection_``
        # delegation. To satisfy our Python base's invariants we still
        # need a valid exercise_time / exercise_date / day_counter.
        # We piggy-back on the base's values.
        ed: Date | None
        rd: Date | None
        dc: DayCounter | None
        try:
            ed = base.exercise_date()
        except Exception:
            ed = None
        try:
            rd = base.reference_date()
        except Exception:
            rd = None
        try:
            dc = base.day_counter()
        except Exception:
            dc = None
        super().__init__(
            exercise_date=ed,
            exercise_time=None if ed is not None and rd is not None else base.exercise_time(),
            day_counter=dc,
            reference_date=rd,
            volatility_type=base.volatility_type(),
            shift=base.shift(),
        )
        self._base: SmileSection = base
        self._spread: Quote = vol_spread
        # Register *this section* as an observer of both the base smile
        # and the spread Quote — when either fires its notify_observers,
        # this section's update() will run, which in turn re-notifies
        # any observer registered with us. Note: register_with(obs)
        # means "obs is registered as listener of self".
        base.register_with(self)
        vol_spread.register_with(self)
        # Pin a strong reference on the base + spread to keep them alive
        # past their weakref-only observer registration. Without this,
        # SimpleQuote.set_value cannot reach this section because the
        # WeakSet entry is reaped.
        # (No-op: ``self._base`` / ``self._spread`` already pin them.)

    # --- delegating overrides -----------------------------------------

    def min_strike(self) -> float:
        return self._base.min_strike()

    def max_strike(self) -> float:
        return self._base.max_strike()

    def atm_level(self) -> float:
        return self._base.atm_level()

    def exercise_time(self) -> float:
        return self._base.exercise_time()

    def exercise_date(self) -> Date:
        return self._base.exercise_date()

    def reference_date(self) -> Date:
        return self._base.reference_date()

    def day_counter(self) -> DayCounter:
        return self._base.day_counter()

    def volatility_type(self) -> VolatilityType:
        return self._base.volatility_type()

    def shift(self) -> float:
        return self._base.shift()

    # --- core -----------------------------------------------------------

    def _volatility_impl(self, strike: float) -> float:
        v = self._base.volatility(strike) + self._spread.value()
        # Vol may not be negative; pin to zero if the spread drives it below.
        return max(v, 0.0)

    def _variance_impl(self, strike: float) -> float:
        v = self._volatility_impl(strike)
        t = self._base.exercise_time()
        return v * v * t

    def update(self) -> None:
        # The C++ class only calls notifyObservers; we mirror that.
        self.notify_observers()
