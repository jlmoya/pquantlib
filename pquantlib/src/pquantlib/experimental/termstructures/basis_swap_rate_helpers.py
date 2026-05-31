"""Basis-swap rate helpers — ibor/ibor + overnight/ibor tenor-basis swaps.

# C++ parity: ql/experimental/termstructures/basisswapratehelpers.{hpp,cpp}
# (v1.42.1).

Two helpers for bootstrapping a forecast curve from quoted tenor-basis
swaps. In both the swap pays ``baseIndex + basis`` and receives
``otherIndex``; an exogenous discount curve is required.

* :class:`IborIborBasisSwapRateHelper` — both legs are ibor. Choose
  which forecast curve to bootstrap via ``bootstrap_base_curve``: when
  True the base-index curve is solved-for (the other index needs an
  existing forecast curve); when False the other-index curve is
  solved-for.
* :class:`OvernightIborBasisSwapRateHelper` — the base leg is overnight,
  the other leg ibor. Bootstraps the ibor (other) forecast curve; the
  overnight (base) index needs an existing forecast curve. If no discount
  curve is supplied, the bootstrapped curve discounts both legs.

``implied_quote`` reproduces the C++ ``-(NPV / legBPS(0)) * 1e-4``: the
basis (in absolute rate, e.g. 0.0010) that zeroes the swap NPV.

The Python port follows the existing ``SwapRateHelper`` idiom — it
rebuilds the swap inside ``implied_quote`` using the currently-set term
structure (cloning the bootstrapped index onto that curve), rather than
threading a C++ ``RelinkableHandle``. The pillar/earliest/latest dates
are computed in ``_initialize_dates`` from the global evaluation date
(via ``RelativeDateBootstrapHelper``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from pquantlib import qassert
from pquantlib.cashflows.coupon_pricer import IborCouponPricer, set_coupon_pricer
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.cashflows.overnight_leg import overnight_leg
from pquantlib.instruments.swap import Swap
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.termstructures.bootstrap_helper import (
    PillarChoice,
    RelativeDateBootstrapHelper,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.schedule import MakeSchedule
from pquantlib.time.time_unit import TimeUnit

_DAYS = TimeUnit.Days

if TYPE_CHECKING:
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.indexes.overnight_index import OvernightIndex
    from pquantlib.quotes.quote import Quote
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.date import Date
    from pquantlib.time.period import Period


def _fixing_end_date(coupon: FloatingRateCoupon, index: IborIndex) -> Date:
    """``index.maturityDate(index.valueDate(coupon.fixingDate()))``.

    # C++ parity: ``IborCoupon::fixingEndDate()`` — the date the index's
    # forward period ends, used for the helper's latest-relevant date.
    """
    fixing_date: Date = coupon.fixing_date()
    return index.maturity_date(index.value_date(fixing_date))


@final
class IborIborBasisSwapRateHelper(RelativeDateBootstrapHelper["YieldTermStructureProtocol"]):
    """Rate helper bootstrapping over ibor-ibor basis swaps.

    # C++ parity: ``class IborIborBasisSwapRateHelper :
    # public RelativeDateRateHelper`` in basisswapratehelpers.hpp:41-75.
    """

    def __init__(
        self,
        basis: Quote | float,
        tenor: Period,
        settlement_days: int,
        calendar: Calendar,
        convention: BusinessDayConvention,
        end_of_month: bool,
        base_index: IborIndex,
        other_index: IborIndex,
        discount_handle: YieldTermStructureProtocol,
        bootstrap_base_curve: bool,
    ) -> None:
        super().__init__(basis)
        self._tenor: Period = tenor
        self._settlement_days: int = settlement_days
        self._calendar: Calendar = calendar
        self._convention: BusinessDayConvention = convention
        self._end_of_month: bool = end_of_month
        self._base_index: IborIndex = base_index
        self._other_index: IborIndex = other_index
        self._discount_handle: YieldTermStructureProtocol = discount_handle
        self._bootstrap_base_curve: bool = bootstrap_base_curve
        self._swap: Swap | None = None
        self._initialize_dates()

    def _build(self, ts: YieldTermStructureProtocol | None) -> Swap:
        # C++ parity: basisswapratehelpers.cpp:63-95 — build the two ibor
        # legs (100 notionals) and a Swap priced off the discount handle.
        # The leg whose forecast curve we bootstrap is cloned onto ``ts``.
        if ts is not None and self._bootstrap_base_curve:
            base_index = self._base_index.clone(ts)
            other_index = self._other_index
        elif ts is not None:
            base_index = self._base_index
            other_index = self._other_index.clone(ts)
        else:
            base_index = self._base_index
            other_index = self._other_index

        today = self._eval_date()
        earliest = self._calendar.advance(
            today, self._settlement_days, _DAYS, BusinessDayConvention.Following
        )
        maturity = self._calendar.advance(
            earliest, self._tenor.length, self._tenor.units, self._convention
        )

        base_schedule = (
            MakeSchedule()
            .from_date(earliest)
            .to(maturity)
            .with_tenor(base_index.tenor())
            .with_calendar(self._calendar)
            .with_convention(self._convention)
            .with_end_of_month(self._end_of_month)
            .forwards()
            .build()
        )
        base_leg = ibor_leg(base_schedule, base_index, [100.0])

        other_schedule = (
            MakeSchedule()
            .from_date(earliest)
            .to(maturity)
            .with_tenor(other_index.tenor())
            .with_calendar(self._calendar)
            .with_convention(self._convention)
            .with_end_of_month(self._end_of_month)
            .forwards()
            .build()
        )
        other_leg = ibor_leg(other_schedule, other_index, [100.0])
        # Both legs are ibor; attach the plain-IBOR forecast pricer (C++ relies
        # on the global setCouponPricer registry — see VanillaSwap divergence).
        set_coupon_pricer(base_leg, IborCouponPricer())
        set_coupon_pricer(other_leg, IborCouponPricer())

        swap = Swap.from_legs(base_leg, other_leg)
        swap.set_pricing_engine(DiscountingSwapEngine(self._discount_handle))
        return swap

    def _initialize_dates(self) -> None:
        # C++ parity: basisswapratehelpers.cpp:63-95 (initializeDates).
        swap = self._build(None)
        self._swap = swap
        today = self._eval_date()
        self._earliest_date = self._calendar.advance(
            today, self._settlement_days, _DAYS, BusinessDayConvention.Following
        )
        self._maturity_date = self._calendar.advance(
            self._earliest_date, self._tenor.length, self._tenor.units, self._convention
        )
        base_last = swap.leg(0)[-1]
        other_last = swap.leg(1)[-1]
        assert isinstance(base_last, FloatingRateCoupon)
        assert isinstance(other_last, FloatingRateCoupon)
        self._latest_relevant_date = max(
            self._maturity_date,
            _fixing_end_date(base_last, self._base_index),
            _fixing_end_date(other_last, self._other_index),
        )
        self._pillar_date = self._latest_relevant_date
        self._latest_date = self._pillar_date
        self._pillar_choice = PillarChoice.LastRelevantDate

    def implied_quote(self) -> float:
        # C++ parity: basisswapratehelpers.cpp:106-109 —
        # -(NPV / legBPS(0)) * 1e-4.
        qassert.require(
            self._term_structure is not None,
            "IborIborBasisSwapRateHelper: term structure not set",
        )
        swap = self._build(self._term_structure)
        return -(swap.npv() / swap.leg_bps(0)) * 1.0e-4

    def swap(self) -> Swap:
        """The underlying basis swap (built at the current evaluation date)."""
        assert self._swap is not None
        return self._swap

    def _eval_date(self) -> Date:
        from pquantlib.patterns.observable_settings import (  # noqa: PLC0415
            ObservableSettings,
        )

        return ObservableSettings().evaluation_date_or_today()


@final
class OvernightIborBasisSwapRateHelper(
    RelativeDateBootstrapHelper["YieldTermStructureProtocol"]
):
    """Rate helper bootstrapping over overnight-ibor basis swaps.

    # C++ parity: ``class OvernightIborBasisSwapRateHelper :
    # public RelativeDateRateHelper`` in basisswapratehelpers.hpp:85-117.
    """

    def __init__(
        self,
        basis: Quote | float,
        tenor: Period,
        settlement_days: int,
        calendar: Calendar,
        convention: BusinessDayConvention,
        end_of_month: bool,
        base_index: OvernightIndex,
        other_index: IborIndex,
        discount_handle: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(basis)
        self._tenor: Period = tenor
        self._settlement_days: int = settlement_days
        self._calendar: Calendar = calendar
        self._convention: BusinessDayConvention = convention
        self._end_of_month: bool = end_of_month
        self._base_index: OvernightIndex = base_index
        self._other_index: IborIndex = other_index
        self._discount_handle: YieldTermStructureProtocol | None = discount_handle
        self._swap: Swap | None = None
        self._initialize_dates()

    def _build(self, ts: YieldTermStructureProtocol | None) -> Swap:
        # C++ parity: basisswapratehelpers.cpp:148-172.
        # Bootstraps the ibor (other) curve; the overnight (base) index keeps
        # its existing forecast curve.
        other_index = self._other_index.clone(ts) if ts is not None else self._other_index
        base_index = self._base_index

        today = self._eval_date()
        earliest = self._calendar.advance(
            today, self._settlement_days, _DAYS, BusinessDayConvention.Following
        )
        maturity = self._calendar.advance(
            earliest, self._tenor.length, self._tenor.units, self._convention
        )

        schedule = (
            MakeSchedule()
            .from_date(earliest)
            .to(maturity)
            .with_tenor(other_index.tenor())
            .with_calendar(self._calendar)
            .with_convention(self._convention)
            .with_end_of_month(self._end_of_month)
            .forwards()
            .build()
        )
        base_leg = overnight_leg(schedule, base_index, [100.0])
        other_leg = ibor_leg(schedule, other_index, [100.0])
        # The ibor leg needs the plain-IBOR pricer; overnight coupons attach
        # their pricer at construction.
        set_coupon_pricer(other_leg, IborCouponPricer())

        # When no exogenous discount curve is supplied, discount off the
        # bootstrapped curve (C++ uses termStructureHandle_ in that case).
        discount = (
            self._discount_handle
            if self._discount_handle is not None
            else ts
        )
        swap = Swap.from_legs(base_leg, other_leg)
        if discount is not None:
            swap.set_pricing_engine(DiscountingSwapEngine(discount))
        return swap

    def _initialize_dates(self) -> None:
        # C++ parity: basisswapratehelpers.cpp:148-172 (initializeDates).
        today = self._eval_date()
        self._earliest_date = self._calendar.advance(
            today, self._settlement_days, _DAYS, BusinessDayConvention.Following
        )
        self._maturity_date = self._calendar.advance(
            self._earliest_date, self._tenor.length, self._tenor.units, self._convention
        )
        # Build with the discount curve if present (else legs only need dates).
        if self._discount_handle is not None:
            swap = self._build(None)
            self._swap = swap
            other_last = swap.leg(1)[-1]
            assert isinstance(other_last, FloatingRateCoupon)
            self._latest_relevant_date = max(
                self._maturity_date, _fixing_end_date(other_last, self._other_index)
            )
        else:
            self._latest_relevant_date = self._maturity_date
        self._pillar_date = self._latest_relevant_date
        self._latest_date = self._pillar_date
        self._pillar_choice = PillarChoice.LastRelevantDate

    def implied_quote(self) -> float:
        # C++ parity: basisswapratehelpers.cpp:185-188.
        qassert.require(
            self._term_structure is not None,
            "OvernightIborBasisSwapRateHelper: term structure not set",
        )
        swap = self._build(self._term_structure)
        return -(swap.npv() / swap.leg_bps(0)) * 1.0e-4

    def swap(self) -> Swap:
        """The underlying basis swap."""
        assert self._swap is not None
        return self._swap

    def _eval_date(self) -> Date:
        from pquantlib.patterns.observable_settings import (  # noqa: PLC0415
            ObservableSettings,
        )

        return ObservableSettings().evaluation_date_or_today()


__all__ = ["IborIborBasisSwapRateHelper", "OvernightIborBasisSwapRateHelper"]
