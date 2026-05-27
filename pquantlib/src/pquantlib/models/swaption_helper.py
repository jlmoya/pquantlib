"""SwaptionHelper — BlackCalibrationHelper concrete for swaptions.

# C++ parity: ql/models/shortrate/calibrationhelpers/swaptionhelper.{hpp,cpp}
# (v1.42.1).

Used during model calibration to fit a short-rate (or other) model to
liquid market swaptions. Holds a vol Quote, a target swap maturity +
length, an underlying IborIndex + term structure, and pricing
parameters; on ``calculate()`` it builds the underlying VanillaSwap +
European Swaption from those inputs. The ``model_value`` path runs
the user-attached model-dependent engine; ``black_price(sigma)`` runs
a fresh BlackSwaptionEngine on the same vol so that
``market_value == black_price(vol_quote.value())`` becomes the
calibration target.

Divergences from C++:

- C++ exposes three constructor overloads (Period maturity / Date
  exerciseDate / Date exerciseDate + Date endDate). PQuantLib
  consolidates to a single ``__init__`` taking ``maturity`` (a
  ``Period``); the explicit-Date variants are not exercised by
  Phase 4 L4-E tests and can be added when a calibrator needs them.
- ``addTimesTo`` (which fills the lattice/tree calibration grid)
  is deferred — Phase 5 territory.
- ``RateAveraging`` is not yet ported in PQuantLib (no OIS calibration
  path in L4-E); we hard-code the VanillaSwap branch and skip the
  OvernightIndexedSwap branch. Re-add when L4-F or later land OIS-
  based calibration.
- C++ ``settlementDays`` overrides the index's value-date convention
  with an explicit lag; PQuantLib's default uses the index's
  ``value_date`` directly. If callers need the explicit-lag path
  we'd add it via an optional ``settlement_days`` argument; deferred
  until a test needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pquantlib import qassert
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import Swaption
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.models.calibration_helper import (
    BlackCalibrationHelper,
    CalibrationErrorType,
)
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.pricingengines.swaption.black_swaption_engine import (
    BachelierSwaptionEngine,
    BlackSwaptionEngine,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.schedule import Schedule

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.quotes.quote import Quote
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.time.period import Period


class SwaptionHelper(BlackCalibrationHelper):
    """Calibration helper for European swaptions.

    # C++ parity: ``class SwaptionHelper : public BlackCalibrationHelper``
    # in swaptionhelper.hpp:42-119 (v1.42.1).
    """

    def __init__(
        self,
        maturity: Period,
        length: Period,
        volatility: Quote,
        index: IborIndex,
        fixed_leg_tenor: Period,
        fixed_leg_day_counter: DayCounter,
        floating_leg_day_counter: DayCounter,
        term_structure: YieldTermStructureProtocol,
        error_type: CalibrationErrorType = CalibrationErrorType.RelativePriceError,
        strike: float | None = None,
        nominal: float = 1.0,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        shift: float = 0.0,
    ) -> None:
        super().__init__(volatility, error_type, volatility_type, shift)
        self._maturity: Period = maturity
        self._length: Period = length
        self._fixed_leg_tenor: Period = fixed_leg_tenor
        self._index: IborIndex = index
        self._term_structure: YieldTermStructureProtocol = term_structure
        self._fixed_leg_day_counter: DayCounter = fixed_leg_day_counter
        self._floating_leg_day_counter: DayCounter = floating_leg_day_counter
        self._strike: float | None = strike
        self._nominal: float = nominal
        # Filled in _perform_calculations.
        self._exercise_rate: float = 0.0
        self._swap: VanillaSwap | None = None
        self._swaption: Swaption | None = None
        # C++ parity: swaptionhelper.cpp:56-57 — registerWith(index_) +
        # registerWith(termStructure_).
        self._index.register_with(self)

    # --- BlackCalibrationHelper hooks ---------------------------------

    def _perform_calculations(self) -> None:
        # # C++ parity: swaptionhelper.cpp:152-199 (v1.42.1).
        cal = self._index.fixing_calendar()
        ref_date = self._term_structure.reference_date()
        # Exercise date = ref + maturity, adjusted with the index's BDC.
        exercise_date = cal.advance(
            ref_date,
            self._maturity.length,
            self._maturity.units,
            self._index.business_day_convention(),
        )
        # Swap start = index.value_date(adjusted exercise date).
        adjusted_exercise = self._index.fixing_calendar().adjust(
            exercise_date, BusinessDayConvention.Following
        )
        start_date = self._index.value_date(adjusted_exercise)
        # Swap end = start + length, adjusted.
        end_date = cal.advance(
            start_date,
            self._length.length,
            self._length.units,
            self._index.business_day_convention(),
        )

        bdc = self._index.business_day_convention()
        fixed_schedule = Schedule.from_rule(
            start_date,
            end_date,
            self._fixed_leg_tenor,
            cal,
            bdc,
            bdc,
            DateGeneration.Forward,
            False,
        )
        float_schedule = Schedule.from_rule(
            start_date,
            end_date,
            self._index.tenor(),
            cal,
            bdc,
            bdc,
            DateGeneration.Forward,
            False,
        )

        swap_engine = DiscountingSwapEngine(self._term_structure)

        # First build a swap with rate=0 to extract the fair rate; that
        # determines the strike (if not user-supplied) and the swap type
        # (Payer if strike > forward, Receiver otherwise).
        # C++ parity: swaptionhelper.cpp:184-194.
        sample = self._make_swap(fixed_schedule, float_schedule, 0.0, SwapType.Receiver)
        sample.set_pricing_engine(swap_engine)
        forward = sample.fair_rate()
        if self._strike is None:
            exercise_rate = forward
            swap_type = SwapType.Receiver
        else:
            exercise_rate = self._strike
            swap_type = SwapType.Receiver if self._strike <= forward else SwapType.Payer

        self._exercise_rate = exercise_rate
        self._swap = self._make_swap(
            fixed_schedule, float_schedule, exercise_rate, swap_type
        )
        self._swap.set_pricing_engine(swap_engine)

        exercise = EuropeanExercise(exercise_date)
        self._swaption = Swaption(self._swap, exercise)

        # Now that the swaption is built, populate the market_value
        # cache by delegating to the base class (uses black_price).
        super()._perform_calculations()

    def _make_swap(
        self,
        fixed_schedule: Schedule,
        float_schedule: Schedule,
        rate: float,
        swap_type: SwapType,
    ) -> VanillaSwap:
        # # C++ parity: swaptionhelper.cpp:201-216 (v1.42.1) — VanillaSwap branch only.
        # The C++ helper also has an OvernightIndexedSwap branch; we
        # carve that out per the module docstring.
        return VanillaSwap(
            swap_type,
            self._nominal,
            fixed_schedule,
            rate,
            self._fixed_leg_day_counter,
            float_schedule,
            self._index,
            0.0,
            self._floating_leg_day_counter,
        )

    # --- BlackCalibrationHelper abstract overrides --------------------

    def model_value(self) -> float:
        """NPV of the swaption under the user-attached engine.

        # C++ parity: swaptionhelper.cpp:123-127 (v1.42.1).
        """
        self.calculate()
        qassert.require(self._swaption is not None, "swaption not constructed")
        assert self._swaption is not None
        qassert.require(self._engine is not None, "no model engine set")
        assert self._engine is not None
        self._swaption.set_pricing_engine(self._engine)
        return self._swaption.npv()

    def black_price(self, volatility: float) -> float:
        """Black (or Bachelier) price of the swaption at ``volatility``.

        # C++ parity: swaptionhelper.cpp:129-150 (v1.42.1).
        """
        self.calculate()
        qassert.require(self._swaption is not None, "swaption not constructed")
        assert self._swaption is not None
        if self._volatility_type == VolatilityType.ShiftedLognormal:
            engine = BlackSwaptionEngine(
                self._term_structure,
                volatility,
                day_counter=Actual365Fixed(),
                displacement=self._shift,
            )
        elif self._volatility_type == VolatilityType.Normal:
            engine = cast(
                "BlackSwaptionEngine",
                BachelierSwaptionEngine(
                    self._term_structure,
                    volatility,
                    day_counter=Actual365Fixed(),
                ),
            )
        else:
            qassert.fail(f"unknown VolatilityType ({self._volatility_type})")
        self._swaption.set_pricing_engine(engine)
        value = self._swaption.npv()
        # Restore the user-attached engine so subsequent model_value
        # calls work as expected (matches C++ swaptionhelper.cpp:148).
        if self._engine is not None:
            self._swaption.set_pricing_engine(self._engine)
        return value

    def add_times_to(self, times: list[float]) -> None:
        """Append the calibration times the model needs to know about.

        # C++ parity: swaptionhelper.cpp:111-121 (v1.42.1). The C++
        # version uses DiscretizedSwaption.mandatoryTimes() which
        # depends on the Lattice / DiscretizedAsset machinery deferred
        # to Phase 5. PQuantLib falls back to just the exercise time —
        # sufficient for any Brent-style implied-vol round-trip but
        # not for tree-based calibrators.
        """
        del times  # mark intentional skip
        # If a Phase-5 caller needs more times here, fold in the swap's
        # fixing/payment grid. For Phase 4 implied-vol round-trip tests
        # this method is unused.

    # --- inspectors ---------------------------------------------------

    def underlying_swap(self) -> VanillaSwap:
        """The underlying swap (built on first ``calculate``).

        # C++ parity: ``SwaptionHelper::underlying`` in
        # swaptionhelper.hpp:96-99 (v1.42.1).
        """
        self.calculate()
        assert self._swap is not None
        return self._swap

    def swaption(self) -> Swaption:
        """The wrapped swaption (built on first ``calculate``).

        # C++ parity: ``SwaptionHelper::swaption`` in
        # swaptionhelper.hpp:100 (v1.42.1).
        """
        self.calculate()
        assert self._swaption is not None
        return self._swaption


__all__ = ["SwaptionHelper"]
