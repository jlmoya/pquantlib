"""AtmAdjustedSmileSection — base smile recentered around a target ATM.

# C++ parity: ql/termstructures/volatility/atmadjustedsmilesection.{hpp,cpp}
# (v1.42.1).

Wraps an existing :class:`SmileSection` and overrides ``atm_level()`` to
return a caller-supplied target. When ``recenter_smile=True`` the
strike axis is shifted by ``adjustment = base.atm_level() - target``
before each base-smile evaluation, so the smile sits at the new ATM
while preserving its shape.

When ``recenter_smile=False`` (the C++ default) the adjustment is zero
and the class behaves identically to :class:`AtmSmileSection`.
"""

from __future__ import annotations

import math

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date


class AtmAdjustedSmileSection(SmileSection):
    """Smile section recentered around a target ATM level.

    Args:
        base: underlying :class:`SmileSection`.
        atm: target ATM level. If ``None`` (or NaN), the base's
            ``atm_level()`` is used.
        recenter_smile: if ``True`` (default ``False`` to match C++)
            shift the strike axis by the difference between the base
            ATM and the target so the curve sits at the new ATM while
            preserving shape.
    """

    def __init__(
        self,
        *,
        base: SmileSection,
        atm: float | None = None,
        recenter_smile: bool = False,
    ) -> None:
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

        # # C++ parity: ``f_ = atm; if (f_ == Null<Real>()) f_ = source_->atmLevel();``
        if atm is None or math.isnan(atm):
            self._atm: float = base.atm_level()
        else:
            self._atm = atm

        # # C++ parity: when ``recenterSmile`` is true and both ATM
        # levels are non-NaN, ``adjustment_ = source.atmLevel() - f_``.
        # Otherwise the adjustment is zero.
        base_atm = base.atm_level()
        if recenter_smile and not math.isnan(self._atm) and not math.isnan(base_atm):
            self._adjustment: float = base_atm - self._atm
        else:
            self._adjustment = 0.0

        self._base.register_with(self)

    # --- forwarded inspectors -----------------------------------------

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

    def adjustment(self) -> float:
        return self._adjustment

    # --- core ---------------------------------------------------------

    def _adjusted_strike(self, strike: float) -> float:
        return strike + self._adjustment

    def _volatility_impl(self, strike: float) -> float:
        return self._base.volatility(self._adjusted_strike(strike))

    def _variance_impl(self, strike: float) -> float:
        return self._base.variance(self._adjusted_strike(strike))

    # # C++ parity: the optionPrice/digitalOptionPrice/density helpers
    # all route strike through ``adjustedStrike(strike)``.

    def option_price(
        self,
        strike: float,
        option_type: int = 1,
        discount: float = 1.0,
    ) -> float:
        return self._base.option_price(self._adjusted_strike(strike), option_type, discount)

    def digital_option_price(
        self,
        strike: float,
        option_type: int = 1,
        discount: float = 1.0,
        gap: float = 1.0e-5,
    ) -> float:
        return self._base.digital_option_price(
            self._adjusted_strike(strike), option_type, discount, gap,
        )

    def density(self, strike: float, discount: float = 1.0, gap: float = 1.0e-4) -> float:
        return self._base.density(self._adjusted_strike(strike), discount, gap)

    def update(self) -> None:
        self.notify_observers()


__all__ = ["AtmAdjustedSmileSection"]
