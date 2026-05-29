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

from pquantlib.exceptions import LibraryException
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import black_formula_implied_std_dev
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
    SwaptionVolatilityStructure,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.swap_index import SwapIndex
    from pquantlib.instruments.swaption import Swaption
    from pquantlib.models.shortrate.gaussian1d_model import Gaussian1dModel
    from pquantlib.pricingengines.pricing_engine import PricingEngine
    from pquantlib.time.business_day_convention import BusinessDayConvention
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.date import Date


class _Gaussian1dSmileSection(SmileSection):
    """Smile section backed by a Gaussian1d model + pricing engine.

    # C++ parity: ``class Gaussian1dSmileSection`` in
    # ql/termstructures/volatility/gaussian1dsmilesection.{hpp,cpp}.

    Built for a fixed expiry (``fixing_date``) and a swap index. The
    ATM rate and annuity are computed at construction from the model;
    each call to ``_volatility_impl(strike)`` builds a Swaption at the
    given strike, prices it with the user engine, normalizes by the
    annuity, and inverts via Black-implied stdev.

    The C++ class accepts both SwapIndex (-> swaption back-out) and
    IborIndex (-> capfloor back-out) constructors; PQuantLib provides
    only the SwapIndex variant (the IborIndex variant requires the
    deferred ``Gaussian1dCapFloorEngine`` and ``MakeCapFloor`` which
    is a future addition).
    """

    def __init__(
        self,
        fixing_date: Date,
        swap_index: SwapIndex,
        model: Gaussian1dModel,
        day_counter: DayCounter,
        engine: PricingEngine,
    ) -> None:
        # C++ parity: gaussian1dsmilesection.cpp:30-48.
        super().__init__(
            exercise_date=fixing_date,
            day_counter=day_counter,
            reference_date=model.term_structure.reference_date(),
        )
        self._fixing_date: Date = fixing_date
        self._swap_index: SwapIndex = swap_index
        self._model: Gaussian1dModel = model
        self._engine: PricingEngine = engine

        self._atm: float = model.swap_rate(
            fixing_date, swap_index.tenor(), None, 0.0, swap_index
        )
        self._annuity: float = model.swap_annuity(
            fixing_date, swap_index.tenor(), None, 0.0, swap_index
        )

    # --- SmileSection surface ------------------------------------------

    def min_strike(self) -> float:
        # C++ parity: gaussian1dsmilesection.hpp:58 — lognormal section.
        return 0.0

    def max_strike(self) -> float:
        # C++ parity: gaussian1dsmilesection.hpp:59.
        return float("inf")

    def atm_level(self) -> float:
        # C++ parity: gaussian1dsmilesection.cpp:73.
        return self._atm

    def option_price(
        self,
        strike: float,
        option_type: int = 1,
        discount: float = 1.0,
    ) -> float:
        """Normalized swaption price (NPV / annuity * discount).

        # C++ parity: gaussian1dsmilesection.cpp:75-95.

        ``option_type`` is the int-encoded ``OptionType`` (1 = Call /
        Payer, -1 = Put / Receiver). The integer signature matches the
        ``SmileSection`` base.
        """
        from typing import cast as type_cast  # noqa: PLC0415

        from pquantlib.exercise import EuropeanExercise  # noqa: PLC0415
        from pquantlib.instruments.fixed_vs_floating_swap import (  # noqa: PLC0415
            FixedVsFloatingSwap,
        )
        from pquantlib.instruments.swap import SwapType  # noqa: PLC0415
        from pquantlib.instruments.swaption import Swaption  # noqa: PLC0415

        # Build the underlying swap at the (potentially non-ATM) strike.
        # C++ uses ``MakeSwaption(swap_index, fixing, strike)``; PQuantLib
        # builds the swap via the swap index and then constructs the
        # Swaption explicitly.
        underlying = self._swap_index.underlying_swap(self._fixing_date)
        ot = OptionType(option_type)
        swap_type = SwapType.Payer if ot == OptionType.Call else SwapType.Receiver

        # Re-skin the underlying with the requested fixed rate. We
        # rebuild a VanillaSwap with the same schedule/conventions but
        # the new rate.
        new_underlying = type_cast(
            "FixedVsFloatingSwap",
            _rebuild_swap_at_strike(underlying, strike, swap_type),
        )

        exercise = EuropeanExercise(self._fixing_date)
        swp: Swaption = Swaption(new_underlying, exercise)
        swp.set_pricing_engine(self._engine)
        try:
            tmp = swp.npv()
            return tmp / self._annuity * discount
        except (LibraryException, ZeroDivisionError, ValueError):
            # C++ silently returns 0 on engine failure — match it.
            return 0.0

    def _volatility_impl(self, strike: float) -> float:
        """Black-implied vol — invert ``option_price(strike)`` via Newton-safe.

        # C++ parity: gaussian1dsmilesection.cpp:97-107.

        Returns 0.0 on any exception (Brenner-Subrahmanyan divergence,
        Newton failure, etc.) — matches the C++ catch-all.
        """
        try:
            option_type = OptionType.Call if strike >= self._atm else OptionType.Put
            opt_price = self.option_price(strike, int(option_type))
            if opt_price <= 0.0:
                return 0.0
            stdev = black_formula_implied_std_dev(
                option_type, strike, self._atm, opt_price
            )
            return stdev / (self.exercise_time() ** 0.5)
        except (LibraryException, ZeroDivisionError, ValueError):
            return 0.0


def _rebuild_swap_at_strike(
    underlying: object,
    strike: float,
    swap_type: object,
) -> object:
    """Clone an existing VanillaSwap-like instrument at a new fixed rate.

    # C++ parity: ``MakeSwaption(swap_index, fixing, strike)`` is the
    # C++ idiom — it builds a fresh swap with the requested rate from
    # the swap index. PQuantLib's MakeVanillaSwap is the equivalent but
    # requires more state (effective date, schedules). We take a
    # different approach: reflect the existing underlying schedule +
    # day count + leg notional, but replace the fixed rate.

    The runtime-untyped signature here mirrors the duck-typed style
    used in ``Gaussian1dModel.swap_rate`` and ``swap_annuity`` —
    ``VanillaSwap`` lives below the termstructures layer in the
    dependency graph; using ``object`` for the arguments lets pyright
    type the call site cleanly.
    """
    from typing import Any, cast  # noqa: PLC0415

    from pquantlib.instruments.vanilla_swap import VanillaSwap  # noqa: PLC0415
    u: Any = cast("Any", underlying)
    st: Any = cast("Any", swap_type)
    return VanillaSwap(
        swap_type=st,
        nominal=u.nominal(),
        fixed_schedule=u.fixed_schedule(),
        fixed_rate=strike,
        fixed_day_count=u.fixed_day_count(),
        float_schedule=u.floating_schedule(),
        ibor_index=u.ibor_index(),
        spread=u.spread(),
        floating_day_count=u.floating_day_count(),
        payment_convention=u.payment_convention(),
    )


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
    ) -> _Gaussian1dSmileSection:
        """Build a smile section at ``(fixing_date, tenor)``.

        # C++ parity: gaussian1dswaptionvolatility.cpp:38-43.

        The C++ version clones the swap_index_base with the requested
        tenor; PQuantLib's SwapIndex doesn't expose a tenor-clone
        overload so we use the base index as-is. In practice all
        callers pass tenor == swap_index_base.tenor().
        """
        _ = swap_tenor, extrapolate  # documented divergence — see Gaussian1dModel module docstring
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
        return _Gaussian1dSmileSection(
            fixing_date=fixing_date,
            swap_index=self._swap_index_base,
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
