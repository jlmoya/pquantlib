"""Gaussian1dSwaptionVolatility — Black implied vol surface from a Gaussian1d model.

# C++ parity:
# - ql/termstructures/volatility/swaption/gaussian1dswaptionvolatility.{hpp,cpp}
# - ql/termstructures/volatility/gaussian1dsmilesection.{hpp,cpp}
# @ v1.42.1 (099987f0).

Given a ``Gaussian1dModel`` (e.g. ``Gsr``) and a swaption pricing
engine, ``Gaussian1dSwaptionVolatility`` returns the Black-implied
volatility at each ``(expiry, tenor, strike)`` by:

1. Building the ATM swap from the swap index at the expiry.
2. Pricing the OTM swaption with the user-provided engine, normalized
   by the model's swap annuity.
3. Inverting the Black formula (Newton-safe) over that normalized
   price to recover the lognormal vol.

The C++ smile section catches all exceptions and returns ``0.0`` on
inversion failure (gaussian1dsmilesection.cpp:97-107). PQuantLib
mirrors that defensive behavior — calibration loops can tolerate a
0 vol at pathological strikes without halting.

## Divergence from C++

- C++ defaults to ``Gaussian1dSwaptionEngine`` (numerical integration
  on the model's state grid). PQuantLib requires the engine to be
  passed in explicitly — we don't carry ``Gaussian1dSwaptionEngine``
  in this cluster (it's a deferred Phase-10 carve-out). Users can
  pass any ``Swaption``-compatible engine; typical use is
  ``BlackSwaptionEngine`` on a constant-vol surface during calibration.

- C++ ``smileSectionImpl(Time, Time)`` uses a Newton root-find from
  a guess of ``optionTime`` years past the reference date to back out
  the option date. PQuantLib's surface dispatches by Date (matches
  the rest of the project); the Time -> Date conversion uses the
  parent class's ``option_date_from_tenor`` plumbing.

- C++ ``Gaussian1dSwaptionVolatility::DateHelper`` (a 1-D Newton
  inversion helper for finding the Date matching a given option time)
  is omitted — PQuantLib's call sites use Date directly, so no time->date
  inversion is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.termstructures.volatility.gaussian1d_smile_section import (
    Gaussian1dSmileSection,
)
from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
    SwaptionVolatilityStructure,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.swap_index import SwapIndex
    from pquantlib.models.shortrate.gaussian1d_model import Gaussian1dModel
    from pquantlib.pricingengines.pricing_engine import PricingEngine
    from pquantlib.time.business_day_convention import BusinessDayConvention
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.date import Date


class Gaussian1dSwaptionVolatility(SwaptionVolatilityStructure):
    """Swaption-volatility surface implied by a Gaussian1dModel + pricing engine.

    # C++ parity: ``class Gaussian1dSwaptionVolatility`` in
    # ql/termstructures/volatility/swaption/gaussian1dswaptionvolatility.{hpp,cpp}
    # (v1.42.1).

    The constructor signature mirrors C++ but takes the engine
    argument explicitly (the C++ default
    ``Gaussian1dSwaptionEngine(model, 64, 7.0, ...)`` is deferred —
    L10-B carve-out). The volatility surface returns Black-implied vol
    backed out of the swaption NPV from the engine.

    Common usage in calibration:

        engine = BlackSwaptionEngine(discount_curve, initial_vol_quote)
        svol = Gaussian1dSwaptionVolatility(cal, bdc, swap_idx, gsr, dc, engine)
        vol = svol.volatility(expiry_period, swap_tenor, strike)
    """

    def __init__(
        self,
        calendar: Calendar,
        business_day_convention: BusinessDayConvention,
        swap_index_base: SwapIndex,
        day_counter: DayCounter,
        model: Gaussian1dModel,
        swaption_engine: PricingEngine,
    ) -> None:
        # C++ parity: gaussian1dswaptionvolatility.cpp:27-36.
        super().__init__(
            business_day_convention=business_day_convention,
            reference_date=model.term_structure.reference_date(),
            calendar=calendar,
            day_counter=day_counter,
        )
        self._swap_index_base: SwapIndex = swap_index_base
        self._model: Gaussian1dModel = model
        self._engine: PricingEngine = swaption_engine
        # C++ parity: gaussian1dswaptionvolatility.hpp:68 — 100 Years.
        self._max_swap_tenor: Period = Period(100, TimeUnit.Years)

    # --- VolatilityTermStructure surface ------------------------------

    def max_date(self) -> Date:
        from pquantlib.time.date import Date  # noqa: PLC0415

        return Date.max_date()

    def min_strike(self) -> float:
        # C++ parity: gaussian1dswaptionvolatility.hpp:51 — 0.0.
        return 0.0

    def max_strike(self) -> float:
        # C++ parity: gaussian1dswaptionvolatility.hpp:52.
        return float("inf")

    # --- SwaptionVolatilityStructure surface --------------------------

    def max_swap_tenor(self) -> Period:
        # C++ parity: gaussian1dswaptionvolatility.hpp:56.
        return self._max_swap_tenor

    def volatility_type(self) -> VolatilityType:
        # C++ parity: SmileSection-defaulted; gaussian1dsmilesection returns
        # ShiftedLognormal (the Black formula's lognormal branch).
        return VolatilityType.ShiftedLognormal

    def smile_section(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        option_expiry: Period | Date | float,
        swap_tenor: Period | float,
        extrapolate: bool = False,
    ) -> Gaussian1dSmileSection:
        """Build a smile section at ``(fixing_date, tenor)``.

        # C++ parity: gaussian1dswaptionvolatility.cpp:38-43 —
        # ``swapIndexBase_->clone(tenor)``.
        #
        # An earlier revision ignored ``swap_tenor`` entirely, on the grounds
        # that SwapIndex had no tenor-clone overload. It didn't, but the
        # consequence was that asking for a 10y-tenor smile silently returned
        # the base index's tenor. ``SwapIndex.clone_with_tenor`` now exists;
        # a Period tenor is honoured, and a bare Time is rejected rather than
        # quietly dropped (there is no Time -> Period inverse here).
        """
        _ = extrapolate
        # Convert option_expiry to a fixing Date.
        from pquantlib.time.date import Date as _Date  # noqa: PLC0415

        if isinstance(option_expiry, Period):
            fixing_date = self.option_date_from_tenor(option_expiry)
        elif isinstance(option_expiry, _Date):
            fixing_date = option_expiry
        else:
            # Time-overload: not used by the model callers; raise to
            # surface incorrect usage.
            raise TypeError(
                "Gaussian1dSwaptionVolatility.smile_section requires a "
                "Date or Period option_expiry; time-Float overload not "
                "supported."
            )
        if isinstance(swap_tenor, Period):
            swap_index = self._swap_index_base.clone_with_tenor(swap_tenor)
        else:
            raise TypeError(
                "Gaussian1dSwaptionVolatility.smile_section requires a Period "
                "swap_tenor; a bare time cannot be turned back into a tenor."
            )
        return Gaussian1dSmileSection(
            fixing_date=fixing_date,
            swap_index=swap_index,
            model=self._model,
            day_counter=self.day_counter(),
            engine=self._engine,
        )

    def _volatility_impl(
        self,
        option_time: float,
        swap_length: float,
        strike: float,
    ) -> float:
        """Black-implied vol at ``(option_time, swap_length, strike)``.

        # C++ parity: gaussian1dswaptionvolatility.cpp:46-71.

        Time -> Date inversion: the C++ uses a NewtonSafe over a
        DateHelper functor. PQuantLib leverages the parent
        ``SwaptionVolatilityStructure.option_date_from_tenor`` plumbing
        for the floating-time -> Date conversion via a Period round-up
        (consistent with how SwaptionVolatilityMatrix dispatches).
        """
        # Round tenor (in years) up to the nearest month for Period
        # reconstruction. C++ uses Rounding(0)(swap_length * 12.0).
        tenor_months = round(swap_length * 12.0)
        tenor = Period(tenor_months, TimeUnit.Months)
        # Time -> Date via the parent class's calendar-aware advance.
        ref = self.reference_date()
        # Advance ref by ``option_time`` years; round to nearest day.
        time_days = round(option_time * 365.25)
        d = ref + time_days
        d = self._swap_index_base.fixing_calendar().adjust(d)
        section = self.smile_section(d, tenor)
        return section.volatility(strike)
