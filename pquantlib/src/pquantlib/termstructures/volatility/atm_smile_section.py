"""AtmSmileSection — ATM-locked adapter over a base smile.

# C++ parity: ql/termstructures/volatility/atmsmilesection.{hpp,cpp}
# (v1.42.1).

Wraps an existing :class:`SmileSection` and overrides ``atm_level()``
to return a caller-supplied fixed value. Useful when the base smile's
ATM level can't be queried (e.g. it's an interpolated grid without a
forward attached) or when the caller wants to recenter the smile
without modifying the volatility evaluation.

All other accessors (volatility / variance / strike bounds / exercise
time / day counter / reference date / volatility type / shift)
delegate to the base.
"""

from __future__ import annotations

import math

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date


class AtmSmileSection(SmileSection):
    """Smile section that overrides ATM level on top of a base.

    Args:
        base: underlying :class:`SmileSection`.
        atm: explicit ATM level. If ``None`` (or NaN) the base's
            ``atm_level()`` is used at call time.
    """

    def __init__(
        self,
        *,
        base: SmileSection,
        atm: float | None = None,
    ) -> None:
        # We piggy-back on the base's date/time anchoring so the
        # parent invariants (exercise_time/day_counter) are valid.
        # Date-anchored if available, else time-anchored.
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
        # C++: ``f_ = atm; if (f_ == Null<Real>()) f_ = source_->atmLevel();``
        if atm is None or math.isnan(atm):
            self._atm: float = base.atm_level()
        else:
            self._atm = atm
        # Propagate base updates.
        self._base.register_with(self)

    # --- forwarded inspectors ----------------------------------------

    def min_strike(self) -> float:
        return self._base.min_strike()

    def max_strike(self) -> float:
        return self._base.max_strike()

    def atm_level(self) -> float:
        return self._atm

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

    # --- core (delegate) ----------------------------------------------

    def _volatility_impl(self, strike: float) -> float:
        return self._base.volatility(strike)

    def _variance_impl(self, strike: float) -> float:
        return self._base.variance(strike)

    def update(self) -> None:
        self.notify_observers()


__all__ = ["AtmSmileSection"]
