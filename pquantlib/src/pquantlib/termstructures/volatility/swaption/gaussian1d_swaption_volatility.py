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

- C++ defaults its ``swaptionEngine`` argument to
  ``Gaussian1dSwaptionEngine(model, 64, 7.0, true, false)``
  (gaussian1dswaptionvolatility.hpp:44). PQuantLib requires the engine to be
  passed explicitly, so the call site is always visible. An earlier revision
  justified this by saying PQuantLib "doesn't carry
  ``Gaussian1dSwaptionEngine`` in this cluster (a deferred Phase-10
  carve-out)"; that stopped being true —
  ``pquantlib/src/pquantlib/pricingengines/swaption/gaussian1d_swaption_engine.py``
  has it — so the only remaining difference is the explicit argument.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.math.rounding import ClosestRounding
from pquantlib.math.solvers1d.newton_safe import NewtonSafe
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
    from pquantlib.termstructures.term_structure import TermStructure
    from pquantlib.time.business_day_convention import BusinessDayConvention
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.date import Date


class DateHelper:
    """Newton functor inverting ``timeFromReference`` back to a date.

    # C++ parity: ``Gaussian1dSwaptionVolatility::DateHelper``
    # (gaussian1dswaptionvolatility.hpp:70-86).

    C++ declares this in the PRIVATE section of
    ``Gaussian1dSwaptionVolatility``; a private nested class has no Python
    analogue, and hiding it would leave the inversion untestable, so it is
    exposed at module scope under the same name.

    The functor is a piecewise-linear interpolation of
    ``TermStructure::timeFromReference`` through the integer date serials,
    shifted so that its root is the (fractional) serial whose year fraction
    equals ``t``::

        h    = date - floor(date)
        f(x) = h * (T(floor(x) + 1) - t) + (1 - h) * (T(floor(x)) - t)

    ``derivative`` is a FORWARD difference with step ``1e-6``, not the exact
    slope — C++ notes it uses forward differencing "to avoid dates before
    reference date" (hpp:82). The step size is part of the observable
    behaviour, because ``NewtonSafe`` switches between Newton and bisection
    based on it.
    """

    def __init__(self, ts: TermStructure, t: float) -> None:
        # C++ parity: gaussian1dswaptionvolatility.hpp:72.
        self._ts: TermStructure = ts
        self._t: float = t

    def __call__(self, date: float) -> float:
        # C++ parity: gaussian1dswaptionvolatility.hpp:73-80. The C++
        # ``static_cast<Date::serial_type>(date)`` truncates TOWARDS ZERO;
        # serials are positive here, so int() matches.
        from pquantlib.time.date import Date as _Date  # noqa: PLC0415

        serial = int(date)
        t1 = self._ts.time_from_reference(_Date(serial)) - self._t
        t2 = self._ts.time_from_reference(_Date(serial + 1)) - self._t
        h = date - serial
        return h * t2 + (1.0 - h) * t1

    def derivative(self, date: float) -> float:
        """Forward difference with step ``1e-6``.

        # C++ parity: gaussian1dswaptionvolatility.hpp:81-85 — verbatim,
        # including the comment's reason for forward rather than central
        # differencing.
        """
        return (self(date + 1e-6) - self(date)) * 1e6


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

    def _smile_section_from_times(
        self, option_time: float, swap_length: float
    ) -> Gaussian1dSmileSection:
        """Smile section addressed by (time, time) rather than (date, tenor).

        # C++ parity: ``Gaussian1dSwaptionVolatility::smileSectionImpl(Time,
        # Time)`` (gaussian1dswaptionvolatility.cpp:45-59)::
        #
        #     DateHelper hlp(*this, optionTime);
        #     NewtonSafe newton;
        #     Date d(static_cast<Date::serial_type>(newton.solve(
        #         hlp, 0.1,
        #         365.25 * optionTime + referenceDate().serialNumber(), 1.0)));
        #     Period tenor(static_cast<Integer>(Rounding(0)(swapLength*12.0)),
        #                  Months);
        #     d = indexBase_->fixingCalendar().adjust(d);
        #
        # The ``365.25 * optionTime`` expression is only the SOLVER'S GUESS,
        # not the answer. An earlier revision used it directly as the date, on
        # the reasoning that PQuantLib "dispatches by Date, so no inversion is
        # needed". Under Actual/365Fixed the true root is
        # ``ref + 365 * optionTime``, so the guess is off by one day at
        # optionTime = 4 and by five days at optionTime = 20 — pinned per day
        # counter in tests/termstructures/volatility/swaption/
        # test_gaussian1d_date_helper.py against
        # migration-harness/references/v143/ts/datehelper.json.
        """
        from pquantlib.time.date import Date as _Date  # noqa: PLC0415

        helper = DateHelper(self, option_time)
        guess = 365.25 * option_time + float(self.reference_date().serial_number())
        newton = NewtonSafe()
        solved = newton.solve(helper, 0.1, guess, 1.0)
        d = _Date(int(solved))
        # C++ ``Rounding(0)`` defaults to Rounding::Closest (rounding.hpp:75-78),
        # which is round-half-away-from-zero — NOT Python's banker's round().
        tenor_months = int(ClosestRounding(0)(swap_length * 12.0))
        tenor = Period(tenor_months, TimeUnit.Months)
        d = self._swap_index_base.fixing_calendar().adjust(d)
        return self.smile_section(d, tenor)

    def _volatility_impl(
        self,
        option_time: float,
        swap_length: float,
        strike: float,
    ) -> float:
        """Black-implied vol at ``(option_time, swap_length, strike)``.

        # C++ parity: ``volatilityImpl(Time, Time, Rate)``
        # (gaussian1dswaptionvolatility.cpp:67-71) — the smile section from
        # :meth:`_smile_section_from_times`, evaluated at ``strike``.
        """
        return self._smile_section_from_times(
            option_time, swap_length
        ).volatility(strike)
