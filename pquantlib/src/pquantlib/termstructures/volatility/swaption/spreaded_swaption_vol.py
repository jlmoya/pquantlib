"""SpreadedSwaptionVolatility — base + Quote-driven additive spread.

# C++ parity: ql/termstructures/volatility/swaption/spreadedswaptionvol.{hpp,cpp}
# (v1.42.1).

Wraps a base ``SwaptionVolatilityStructure`` and returns
``base.volatility(expiry, tenor, K, true) + spread.value()``
everywhere. Forwards calendar / vol-type / max-tenor / shift to the
base.

C++ ``smileSectionImpl`` returns a ``SpreadedSmileSection``;
PQuantLib defers that with the rest of the smile-section family
(Phase 9 SABR carve-out).
"""

from __future__ import annotations

from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
    SwaptionVolatilityStructure,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class SpreadedSwaptionVolatility(SwaptionVolatilityStructure):
    """Base swaption vol + additive Quote-driven spread."""

    def __init__(
        self,
        base: SwaptionVolatilityStructure,
        spread: Quote,
    ) -> None:
        super().__init__(
            business_day_convention=base.business_day_convention(),
            calendar=base.calendar(),
            day_counter=base.day_counter(),
        )
        self._base: SwaptionVolatilityStructure = base
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

    def max_swap_tenor(self) -> Period:
        return self._base.max_swap_tenor()

    def volatility_type(self) -> VolatilityType:
        return self._base.volatility_type()

    # --- impl -----------------------------------------------------------

    def _volatility_impl(
        self, option_time: float, swap_length: float, strike: float
    ) -> float:
        return (
            self._base.volatility(option_time, swap_length, strike, True)
            + self._spread.value()
        )

    def _shift_impl(self, option_time: float, swap_length: float) -> float:
        return self._base.shift(option_time, swap_length, True)
