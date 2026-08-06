"""MakeSwaption — chained builder for a standard market swaption.

# C++ parity: ql/instruments/makeswaption.{hpp,cpp} (v1.43).

The underlying swap is built from the swap index's own conventions (tenor,
fixing calendar, fixed-leg tenor / convention / day counter), so almost nothing
about the *swap* is configurable here — the setters configure the **option**:
its exercise date, the calendar and convention that derive it, the settlement
type/method, the nominal, the payer/receiver direction and the coupon
convention of the underlying.

Two behaviours are load-bearing and easy to drop:

* the fixing date is computed once and then **cached on the builder**
  (``fixingDate_`` is ``mutable`` in C++, makeswaption.cpp:63-66), so a second
  ``build()`` reuses it even if the evaluation date has moved;
* with no explicit strike the swaption is struck ATM on the index's own
  underlying swap, priced on the index's discount curve when it has an
  exogenous one and on its forecast curve otherwise (makeswaption.cpp:78-107).

Not ported: the ``OvernightIndexedSwapIndex`` branch (makeswaption.cpp:110-124),
which builds the underlying with ``MakeOIS``. PQuantLib has no
``OvernightIndexedSwapIndex`` class yet; when it lands, that branch belongs
here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.make_vanilla_swap import make_vanilla_swap
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import (
    SettlementMethod,
    SettlementType,
    Swaption,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.indexes.swap_index import SwapIndex
    from pquantlib.pricingengines.pricing_engine import PricingEngine
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.date import Date

_ZERO_PERIOD = Period(0, TimeUnit.Days)


class MakeSwaption:
    """Chained builder for :class:`~pquantlib.instruments.swaption.Swaption`.

    # C++ parity: ``MakeSwaption`` (makeswaption.hpp:45-88). Every C++
    # ``withXxx`` setter is present as ``with_xxx`` and returns ``self``;
    # C++'s ``operator ext::shared_ptr<Swaption>()`` is :meth:`build`.
    """

    def __init__(
        self,
        swap_index: SwapIndex,
        option_tenor: Period,
        strike: float | None = None,
    ) -> None:
        # # C++ parity: MakeSwaption(swapIndex, optionTenor, strike)
        # # (makeswaption.cpp:35-42).
        self._swap_index: SwapIndex = swap_index
        self._delivery: SettlementType = SettlementType.Physical
        self._settlement_method: SettlementMethod = SettlementMethod.PhysicalOTC
        self._option_tenor: Period | None = option_tenor
        self._option_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing
        self._fixing_date: Date | None = None
        self._exercise_date: Date | None = None
        self._exercise_calendar: Calendar | None = None
        self._strike: float | None = strike
        self._underlying_type: SwapType = SwapType.Payer
        self._nominal: float = 1.0
        self._use_indexed_coupons: bool | None = None
        self._engine: PricingEngine | None = None

    @classmethod
    def from_fixing_date(
        cls,
        swap_index: SwapIndex,
        fixing_date: Date,
        strike: float | None = None,
    ) -> MakeSwaption:
        """Build against an explicit fixing date instead of an option tenor.

        # C++ parity: the second ``MakeSwaption(swapIndex, fixingDate, strike)``
        # constructor (makeswaption.cpp:44-49). Note that C++ leaves
        # ``nominal_`` **uninitialised-by-default** there — it is a
        # default-initialised ``Real`` member with no in-class initialiser and
        # the ctor's init list omits it. PQuantLib keeps the documented 1.0,
        # which is what the tenor constructor uses; relying on the C++ value is
        # undefined behaviour, not a convention worth porting.
        """
        obj = cls(swap_index, _ZERO_PERIOD, strike)
        # The option tenor is unused on this path — the fixing date is given.
        obj._option_tenor = None
        obj._fixing_date = fixing_date
        return obj

    # --- chained setters ---------------------------------------------------

    def with_nominal(self, n: float) -> MakeSwaption:
        """# C++ parity: ``withNominal`` (makeswaption.cpp:178-181)."""
        self._nominal = n
        return self

    def with_settlement_type(self, delivery: SettlementType) -> MakeSwaption:
        """# C++ parity: ``withSettlementType`` (makeswaption.cpp:144-147)."""
        self._delivery = delivery
        return self

    def with_settlement_method(self, settlement_method: SettlementMethod) -> MakeSwaption:
        """# C++ parity: ``withSettlementMethod`` (makeswaption.cpp:149-153)."""
        self._settlement_method = settlement_method
        return self

    def with_option_convention(self, bdc: BusinessDayConvention) -> MakeSwaption:
        """# C++ parity: ``withOptionConvention`` (makeswaption.cpp:155-159)."""
        self._option_convention = bdc
        return self

    def with_exercise_date(self, date: Date) -> MakeSwaption:
        """# C++ parity: ``withExerciseDate`` (makeswaption.cpp:161-164)."""
        self._exercise_date = date
        return self

    def with_exercise_calendar(self, cal: Calendar) -> MakeSwaption:
        """# C++ parity: ``withExerciseCalendar`` (makeswaption.cpp:166-169)."""
        self._exercise_calendar = cal
        return self

    def with_underlying_type(self, type_: SwapType) -> MakeSwaption:
        """# C++ parity: ``withUnderlyingType`` (makeswaption.cpp:171-174)."""
        self._underlying_type = type_
        return self

    def with_indexed_coupons(self, b: bool | None = True) -> MakeSwaption:
        """# C++ parity: ``withIndexedCoupons`` (makeswaption.cpp:183-186)."""
        self._use_indexed_coupons = b
        return self

    def with_at_par_coupons(self, b: bool = True) -> MakeSwaption:
        """# C++ parity: ``withAtParCoupons`` (makeswaption.cpp:188-191)."""
        self._use_indexed_coupons = not b
        return self

    def with_pricing_engine(self, engine: PricingEngine) -> MakeSwaption:
        """# C++ parity: ``withPricingEngine`` (makeswaption.cpp:176-180)."""
        self._engine = engine
        return self

    # --- construction ------------------------------------------------------

    def build(self) -> Swaption:
        """Build the swaption.

        # C++ parity: ``MakeSwaption::operator ext::shared_ptr<Swaption>()``
        # (makeswaption.cpp:56-142).
        """
        index = self._swap_index
        calendar = self._exercise_calendar if self._exercise_calendar is not None else index.fixing_calendar()
        # If the evaluation date is not a business day then move to the next
        # business day.
        ref_date = calendar.adjust(ObservableSettings().evaluation_date_or_today())
        if self._fixing_date is None:
            assert self._option_tenor is not None
            self._fixing_date = calendar.advance_period(ref_date, self._option_tenor, self._option_convention)
        fixing_date = self._fixing_date

        if self._exercise_date is None:
            exercise = EuropeanExercise(fixing_date)
        else:
            qassert.require(
                self._exercise_date <= fixing_date,
                f"exercise date ({self._exercise_date}) must be less than or "
                f"equal to fixing date ({fixing_date})",
            )
            exercise = EuropeanExercise(self._exercise_date)

        if self._strike is None:
            # ATM on curve(s) attached to index.
            qassert.require(
                index.forwarding_term_structure() is not None,
                f"null term structure set to this instance of {index.name()}",
            )
            temp = index.underlying_swap(fixing_date)
            discount = (
                index.discounting_term_structure()
                if index.exogenous_discount()
                else index.forwarding_term_structure()
            )
            assert discount is not None
            temp.set_pricing_engine(DiscountingSwapEngine(discount, include_settlement_date_flows=False))
            used_strike: float = temp.fair_rate()
        else:
            used_strike = self._strike

        bdc = index.fixed_leg_convention()
        underlying = make_vanilla_swap(
            index.tenor(),
            index.ibor_index(),
            used_strike,
            effective_date=index.value_date(fixing_date),
            fixed_leg_calendar=index.fixing_calendar(),
            # C++ uses ``swapIndex_->dayCounter()``, which for a SwapIndex *is*
            # the fixed-leg day counter it was constructed with.
            fixed_leg_day_count=index.fixed_leg_day_counter(),
            fixed_leg_tenor=index.fixed_leg_tenor(),
            fixed_leg_convention=bdc,
            fixed_leg_termination_convention=bdc,
            swap_type=self._underlying_type,
            nominal=self._nominal,
            use_indexed_coupons=self._use_indexed_coupons,
        )

        swaption = Swaption(underlying, exercise, self._delivery, self._settlement_method)
        if self._engine is not None:
            swaption.set_pricing_engine(self._engine)
        return swaption


__all__ = ["MakeSwaption"]
