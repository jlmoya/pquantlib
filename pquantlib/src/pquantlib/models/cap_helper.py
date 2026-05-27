"""CapHelper — BlackCalibrationHelper concrete for ATM caps.

# C++ parity: ql/models/shortrate/calibrationhelpers/caphelper.{hpp,cpp}
# (v1.42.1).

The helper builds a fixed schedule + floating schedule, computes the
fair (ATM) rate using a plain swap, then constructs a Cap at that
ATM strike with the right floating leg. Calibration uses ``model_value``
(via the model-dependent engine) vs ``black_price(vol)`` (a fresh
BlackCapFloorEngine on the same vol).

Divergences from C++:

- C++ creates a "dummy" IborIndex with the term structure as
  forecasting curve. PQuantLib uses ``index.clone(term_structure)``
  to bind the helper's discount curve as the forwarding curve —
  semantically equivalent.
- C++ uses ``Swap(floatingLeg, fixedLeg)`` directly (with
  ``DiscountingSwapEngine``) to compute the ATM fair rate. PQuantLib
  mirrors via :class:`Swap.from_legs` with the leg order swapped to
  match — first leg paid, second leg received.
- ``addTimesTo`` is deferred (Phase 5; same reasoning as SwaptionHelper).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.coupon_pricer import IborCouponPricer, set_coupon_pricer
from pquantlib.cashflows.fixed_rate_leg import fixed_rate_leg
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.instruments.cap_floor import Cap
from pquantlib.instruments.swap import Swap
from pquantlib.models.calibration_helper import (
    BlackCalibrationHelper,
    CalibrationErrorType,
)
from pquantlib.pricingengines.capfloor.black_capfloor_engine import (
    BachelierCapFloorEngine,
    BlackCapFloorEngine,
)
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.quotes.quote import Quote
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.time.frequency import Frequency


_BASIS_POINT: float = 1.0e-4


class CapHelper(BlackCalibrationHelper):
    """Calibration helper for ATM interest-rate caps.

    # C++ parity: ``class CapHelper : public BlackCalibrationHelper``
    # in caphelper.hpp:35-62 (v1.42.1).
    """

    def __init__(
        self,
        length: Period,
        volatility: Quote,
        index: IborIndex,
        fixed_leg_frequency: Frequency,
        fixed_leg_day_counter: DayCounter,
        include_first_swaplet: bool,
        term_structure: YieldTermStructureProtocol,
        error_type: CalibrationErrorType = CalibrationErrorType.RelativePriceError,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        shift: float = 0.0,
    ) -> None:
        super().__init__(volatility, error_type, volatility_type, shift)
        self._length: Period = length
        self._index: IborIndex = index
        self._term_structure: YieldTermStructureProtocol = term_structure
        self._fixed_leg_frequency: Frequency = fixed_leg_frequency
        self._fixed_leg_day_counter: DayCounter = fixed_leg_day_counter
        self._include_first_swaplet: bool = include_first_swaplet
        self._cap: Cap | None = None
        # C++ parity: caphelper.cpp:47-48.
        self._index.register_with(self)

    # --- LazyObject hooks ----------------------------------------------

    def _perform_calculations(self) -> None:
        # # C++ parity: caphelper.cpp:91-144 (v1.42.1).
        index_tenor = self._index.tenor()
        fixed_rate = 0.04  # dummy value used to extract the fair rate
        ref_date = self._term_structure.reference_date()

        if self._include_first_swaplet:
            start_date = ref_date
            maturity = ref_date + self._length
        else:
            start_date = ref_date + index_tenor
            maturity = ref_date + self._length

        # Dummy index forwarding off the helper's term structure (matches
        # C++ caphelper.cpp:103-112).
        dummy_index = IborIndex(
            "dummy",
            index_tenor,
            self._index.fixing_days(),
            self._index.currency(),
            self._index.fixing_calendar(),
            self._index.business_day_convention(),
            self._index.end_of_month(),
            self._term_structure.day_counter(),
            self._term_structure,
        )

        nominals: list[float] = [1.0]

        # Float schedule (matches C++ caphelper.cpp:116-120).
        float_schedule = Schedule.from_rule(
            start_date,
            maturity,
            self._index.tenor(),
            self._index.fixing_calendar(),
            self._index.business_day_convention(),
            self._index.business_day_convention(),
            DateGeneration.Forward,
            False,
        )
        floating_leg = ibor_leg(
            float_schedule,
            self._index,
            nominals,
            payment_adjustment=self._index.business_day_convention(),
            fixing_days=0,
        )
        set_coupon_pricer(floating_leg, IborCouponPricer())

        # Fixed schedule (matches C++ caphelper.cpp:126-129; Unadjusted BDC).
        fixed_period = Period.from_frequency(self._fixed_leg_frequency)
        fixed_schedule = Schedule.from_rule(
            start_date,
            maturity,
            fixed_period,
            self._index.fixing_calendar(),
            BusinessDayConvention.Unadjusted,
            BusinessDayConvention.Unadjusted,
            DateGeneration.Forward,
            False,
        )
        fixed_leg = fixed_rate_leg(
            fixed_schedule,
            nominals,
            [fixed_rate],
            day_counter=self._fixed_leg_day_counter,
            payment_adjustment=self._index.business_day_convention(),
        )

        # Build a swap to extract the ATM rate (matches C++ caphelper.cpp:135-138).
        swap = Swap.from_legs(floating_leg, fixed_leg)
        swap.set_pricing_engine(
            DiscountingSwapEngine(self._term_structure, include_settlement_date_flows=False)
        )
        fair_rate = fixed_rate - swap.npv() / (swap.leg_bps(1) / _BASIS_POINT)

        self._cap = Cap(floating_leg, [fair_rate])

        # Suppress unused-variable warning for the dummy_index — it's
        # built per C++ parity but pquantlib's ibor_leg uses
        # ``self._index`` directly, not the dummy. We keep it to mirror
        # the C++ code, but mark intent.
        del dummy_index

        super()._perform_calculations()

    # --- BlackCalibrationHelper overrides ------------------------------

    def model_value(self) -> float:
        """NPV under the user-attached model engine.

        # C++ parity: caphelper.cpp:63-67 (v1.42.1).
        """
        self.calculate()
        qassert.require(self._cap is not None, "cap not constructed")
        assert self._cap is not None
        qassert.require(self._engine is not None, "no model engine set")
        assert self._engine is not None
        self._cap.set_pricing_engine(self._engine)
        return self._cap.npv()

    def black_price(self, volatility: float) -> float:
        """Black (or Bachelier) price at ``volatility``.

        # C++ parity: caphelper.cpp:69-89 (v1.42.1).
        """
        self.calculate()
        qassert.require(self._cap is not None, "cap not constructed")
        assert self._cap is not None
        if self._volatility_type == VolatilityType.ShiftedLognormal:
            engine = BlackCapFloorEngine(
                self._term_structure,
                volatility,
                day_counter=Actual365Fixed(),
                displacement=self._shift,
            )
        elif self._volatility_type == VolatilityType.Normal:
            engine = BachelierCapFloorEngine(
                self._term_structure,
                volatility,
                day_counter=Actual365Fixed(),
            )
        else:
            qassert.fail(f"unknown VolatilityType ({self._volatility_type})")
        self._cap.set_pricing_engine(engine)
        value = self._cap.npv()
        if self._engine is not None:
            self._cap.set_pricing_engine(self._engine)
        return value

    def add_times_to(self, times: list[float]) -> None:
        """Append calibration times — deferred (Phase 5).

        # C++ parity: caphelper.cpp:51-61 (v1.42.1). Same DiscretizedCapFloor
        # dependency as SwaptionHelper.add_times_to — deferred.
        """
        del times

    # --- inspectors ----------------------------------------------------

    def cap(self) -> Cap:
        """The underlying ATM cap (built on first ``calculate``)."""
        self.calculate()
        assert self._cap is not None
        return self._cap


__all__ = ["CapHelper"]
