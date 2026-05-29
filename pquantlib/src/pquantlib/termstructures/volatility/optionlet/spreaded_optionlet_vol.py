"""SpreadedOptionletVolatility — base + Quote-driven additive spread.

# C++ parity: ql/termstructures/volatility/optionlet/spreadedoptionletvol.{hpp,cpp}
# (v1.42.1).

Wraps a base ``OptionletVolatilityStructure`` and returns
``base.volatility(t, K, true) + spread.value()`` everywhere. Forwards
date / calendar / convention / vol-type / displacement attributes to
the base.

C++ ``smileSectionImpl`` returns a ``SpreadedSmileSection``;
PQuantLib defers ``SpreadedSmileSection`` along with the rest of the
smile-section family (Phase 9 SABR carve-out).
"""

from __future__ import annotations

from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.optionlet.optionlet_volatility_structure import (
    OptionletVolatilityStructure,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date


class SpreadedOptionletVolatility(OptionletVolatilityStructure):
    """Base optionlet vol + additive Quote-driven spread."""

    def __init__(
        self,
        base: OptionletVolatilityStructure,
        spread: Quote,
    ) -> None:
        # # C++ parity: forwards ``businessDayConvention`` from the base
        # then registers as an observer of both base and spread.
        # PQuantLib's parent ctor needs a reference_date and calendar;
        # we read them from the base.
        super().__init__(
            business_day_convention=base.business_day_convention(),
            calendar=base.calendar(),
            day_counter=base.day_counter(),
        )
        self._base: OptionletVolatilityStructure = base
        self._spread: Quote = spread
        # Forward the base's extrapolation flag.
        self.enable_extrapolation(base.allows_extrapolation())
        self._base.register_with(self)
        self._spread.register_with(self)

    # --- forwarded inspectors -------------------------------------------

    def max_date(self) -> Date:
        return self._base.max_date()

    def reference_date(self) -> Date:
        return self._base.reference_date()

    def min_strike(self) -> float:
        return self._base.min_strike()

    def max_strike(self) -> float:
        return self._base.max_strike()

    def volatility_type(self) -> VolatilityType:
        return self._base.volatility_type()

    def displacement(self) -> float:
        return self._base.displacement()

    # --- impl -----------------------------------------------------------

    def _volatility_impl(self, t: float, strike: float) -> float:
        # # C++ parity: SpreadedOptionletVolatility::volatilityImpl —
        # ``baseVol_->volatility(t, s, true) + spread_->value()``.
        return self._base.volatility(t, strike, True) + self._spread.value()
