"""CPICoupon + CPICashFlow — coupons / cashflows paying a CPI ratio.

# C++ parity: ql/cashflows/cpicoupon.{hpp,cpp} (v1.42.1).

The C++ ``CPICoupon`` provides three overloaded constructors:

1. ``CPICoupon(baseCPI, ...)`` — explicit base CPI (no base date).
2. ``CPICoupon(baseDate, ...)`` — base CPI looked up via the index history
   at ``baseDate``.
3. ``CPICoupon(baseCPI, baseDate, ...)`` — both supplied; the explicit
   ``baseCPI`` wins.

The Python port collapses these to a single constructor that accepts
``base_cpi`` and/or ``base_date`` (both optional, at least one must be
non-null). The dispatch logic + null-check mirrors the C++
``CPICoupon::CPICoupon(Real baseCPI, const Date& baseDate, ...)``
forwarded path.

``CPICashFlow`` ports analogously — it's a single cash flow that pays
``notional * (CPI(observation_date) / baseFixing)`` (optionally minus 1
for swap-style).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows import cash_flow_vectors as cfv
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.indexed_cashflow import IndexedCashFlow
from pquantlib.cashflows.inflation_coupon import InflationCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.inflation.cpi import InterpolationType, lagged_fixing
from pquantlib.indexes.inflation.inflation_index import ZeroInflationIndex
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.cashflows.inflation_coupon_pricer import InflationCouponPricer
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.schedule import Schedule

# Module-level null Date for defaults (avoids B008).
_NULL_DATE: Date = Date()

# Numerical guard mirroring the C++ ``|baseCPI_| < 1e-16`` check.
_CPI_EPSILON: float = 1e-16


class CPICoupon(InflationCoupon):
    """Coupon paying ``nominal * fixedRate * (I_end / I_base) * accrualPeriod``.

    # C++ parity: ``CPICoupon`` in cpicoupon.hpp. The rate is set by a
    # ``CPICouponPricer`` to ``fixedRate * indexRatio(accrualEndDate)``,
    # so the amount works out to the formula in the docstring.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        accrual_start_date: Date,
        accrual_end_date: Date,
        index: ZeroInflationIndex,
        observation_lag: Period,
        observation_interpolation: InterpolationType,
        day_counter: DayCounter,
        fixed_rate: float,
        base_cpi: float | None = None,
        base_date: Date | None = None,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
        ex_coupon_date: Date | None = None,
    ) -> None:
        # C++ parity: ql/cashflows/cpicoupon.cpp:80-91 — the unified
        # constructor takes both baseCPI and baseDate (either may be null,
        # but not both). fixing_days is hard-coded to 0 by C++ at
        # cpicoupon.cpp:81.
        super().__init__(
            payment_date=payment_date,
            nominal=nominal,
            accrual_start_date=accrual_start_date,
            accrual_end_date=accrual_end_date,
            fixing_days=0,
            index=index,
            observation_lag=observation_lag,
            day_counter=day_counter,
            ref_period_start=ref_period_start,
            ref_period_end=ref_period_end,
            ex_coupon_date=ex_coupon_date,
        )
        qassert.require(
            base_cpi is not None or (base_date is not None and base_date != _NULL_DATE),
            "baseCPI and baseDate can not be both null, "
            "provide a valid baseCPI or baseDate",
        )
        if base_cpi is not None:
            qassert.require(
                abs(base_cpi) > _CPI_EPSILON,
                "|baseCPI_| < 1e-16, future divide-by-zero problem",
            )
        self._base_cpi: float | None = base_cpi
        self._fixed_rate: float = fixed_rate
        self._observation_interpolation: InterpolationType = observation_interpolation
        self._base_date: Date = base_date if base_date is not None else _NULL_DATE
        self._cpi_index: ZeroInflationIndex = index

    # ---- inspectors --------------------------------------------------

    def fixed_rate(self) -> float:
        """C++ parity: ql/cashflows/cpicoupon.hpp:259-261 (inline)."""
        return self._fixed_rate

    def base_cpi(self) -> float | None:
        """C++ parity: ql/cashflows/cpicoupon.hpp:271-273 (inline).

        Returns ``None`` (== C++ ``Null<Rate>()``) if only a base date
        was supplied at construction.
        """
        return self._base_cpi

    def base_date(self) -> Date:
        """C++ parity: ql/cashflows/cpicoupon.hpp:275-277 (inline)."""
        return self._base_date

    def observation_interpolation(self) -> InterpolationType:
        """C++ parity: ql/cashflows/cpicoupon.hpp:279-281 (inline)."""
        return self._observation_interpolation

    def cpi_index(self) -> ZeroInflationIndex:
        """C++ parity: ql/cashflows/cpicoupon.hpp:283-285 (inline)."""
        return self._cpi_index

    # ---- InflationCoupon overrides -----------------------------------

    def index_fixing(self) -> float:
        """C++ parity: ql/cashflows/cpicoupon.hpp:267-269 (inline).

        Returns the lagged fixing observed at the accrual end date.
        """
        return lagged_fixing(
            self._cpi_index,
            self._accrual_end_date,
            self._observation_lag,
            self._observation_interpolation,
        )

    def check_pricer_impl(self, pricer: InflationCouponPricer) -> bool:
        """C++ parity: ql/cashflows/cpicoupon.cpp:131-135.

        Accepts only ``CPICouponPricer`` subtypes.
        """
        # Local import to avoid the coupon ↔ pricer cycle.
        from pquantlib.cashflows.cpi_coupon_pricer import CPICouponPricer  # noqa: PLC0415

        return isinstance(pricer, CPICouponPricer)

    def accrued_amount(self, d: Date) -> float:
        """C++ parity: ql/cashflows/cpicoupon.cpp:101-110.

        Uses the pricer's ``accrued_rate(d)`` rather than the abstract
        ``rate()`` (which would discount through the whole accrual
        period). For a CPI coupon, the rate-at-d ≠ rate-at-end except in
        the trivial growth-period-ends-today case.
        """
        if d <= self._accrual_start_date or d > self._payment_date:
            return 0.0
        # Local import for the cycle.
        from pquantlib.cashflows.cpi_coupon_pricer import CPICouponPricer  # noqa: PLC0415

        pricer = self._pricer
        qassert.require(pricer is not None, "pricer not set or of wrong type")
        assert pricer is not None
        qassert.require(
            isinstance(pricer, CPICouponPricer),
            "pricer not set or of wrong type",
        )
        assert isinstance(pricer, CPICouponPricer)
        pricer.initialize(self)
        return self._nominal * pricer.accrued_rate(d) * self.accrued_period(d)

    # ---- CPI-specific accessors --------------------------------------

    def index_ratio(self, d: Date) -> float:
        """``laggedFixing(d) / baseFixing``.

        # C++ parity: ql/cashflows/cpicoupon.cpp:112-129.

        If a base CPI was supplied, divide by it. Otherwise look up the
        base fixing at ``base_date + observation_lag`` with the same
        interpolation.
        """
        i0 = self._base_cpi
        if i0 is None:
            i0 = lagged_fixing(
                self._cpi_index,
                self._base_date + self._observation_lag,
                self._observation_lag,
                self._observation_interpolation,
            )
        i1 = lagged_fixing(
            self._cpi_index,
            d,
            self._observation_lag,
            self._observation_interpolation,
        )
        return i1 / i0

    def adjusted_index_growth(self) -> float:
        """``rate / fixedRate`` — index growth after pricer adjustments.

        # C++ parity: ql/cashflows/cpicoupon.hpp:263-265 (inline).
        """
        return self.rate() / self._fixed_rate


class CPICashFlow(IndexedCashFlow):
    """Single CPI cash flow — ``notional * I(obs) / I(base)`` (or growth-only).

    # C++ parity: ``CPICashFlow`` in cpicoupon.hpp:164-200. Unlike
    # ``ZeroInflationCashFlow``, the base date and base fixing are taken
    # *separately* (so the base fixing can be a quoted/explicit value
    # rather than read from the index history).
    """

    def __init__(
        self,
        notional: float,
        index: ZeroInflationIndex,
        base_date: Date,
        base_fixing: float | None,
        observation_date: Date,
        observation_lag: Period,
        interpolation: InterpolationType,
        payment_date: Date,
        growth_only: bool = False,
    ) -> None:
        # C++ parity: ql/cashflows/cpicoupon.cpp:139-150 — base date is
        # passed as-is, fixing date is ``observationDate - observationLag``.
        super().__init__(
            notional=notional,
            index=index,
            base_date=base_date,
            fixing_date=observation_date - observation_lag,
            payment_date=payment_date,
            growth_only=growth_only,
        )
        qassert.require(
            base_fixing is not None or base_date != _NULL_DATE,
            "baseCPI and baseDate can not be both null, "
            "provide a valid baseCPI or baseDate",
        )
        if base_fixing is not None:
            qassert.require(
                abs(base_fixing) > _CPI_EPSILON,
                "|baseCPI_| < 1e-16, future divide-by-zero problem",
            )
        self._base_fixing: float | None = base_fixing
        self._observation_date: Date = observation_date
        self._observation_lag: Period = observation_lag
        self._interpolation: InterpolationType = interpolation
        self._frequency = index.frequency()
        self._cpi_index: ZeroInflationIndex = index

    # ---- inspectors --------------------------------------------------

    def observation_date(self) -> Date:
        """C++ parity: ql/cashflows/cpicoupon.hpp:182 (inline)."""
        return self._observation_date

    def observation_lag(self) -> Period:
        return self._observation_lag

    def interpolation(self) -> InterpolationType:
        return self._interpolation

    def cpi_index(self) -> ZeroInflationIndex:
        return self._cpi_index

    def base_date(self) -> Date:
        """C++ parity: ql/cashflows/cpicoupon.cpp:159-166 — raises if no
        base date was specified.
        """
        d = self._base_date
        qassert.require(d != _NULL_DATE, "no base date specified")
        return d

    # ---- IndexedCashFlow overrides -----------------------------------

    def base_fixing(self) -> float:
        """C++ parity: ql/cashflows/cpicoupon.cpp:168-173.

        If an explicit base fixing was provided, return it. Otherwise
        look up the lagged fixing at the base date with the configured
        interpolation (and zero lag, since the lag has already been
        consumed by the fact that the base date is *the* base reference).
        """
        if self._base_fixing is not None:
            return self._base_fixing
        return lagged_fixing(
            self._cpi_index,
            self.base_date(),
            Period(0, TimeUnit.Months),
            self._interpolation,
        )

    def index_fixing(self) -> float:
        """C++ parity: ql/cashflows/cpicoupon.cpp:175-177."""
        return lagged_fixing(
            self._cpi_index,
            self._observation_date,
            self._observation_lag,
            self._interpolation,
        )


class CPILeg:
    """Chained builder for a sequence of CPI coupons plus the notional flow.

    # C++ parity: ``CPILeg`` (cpicoupon.hpp:205-247, .cpp:179-350).

    Every C++ ``withXxx`` setter is present as ``with_xxx`` and returns
    ``self``; C++'s ``operator Leg()`` is :meth:`build`. A CPI leg always
    ends with a :class:`CPICashFlow` notional exchange, even when the
    schedule has a single date.
    """

    def __init__(
        self,
        schedule: Schedule,
        index: ZeroInflationIndex,
        base_cpi: float | None,
        observation_lag: Period,
    ) -> None:
        # C++ parity: .cpp:179-186 — the payment day counter defaults to
        # Thirty360(BondBasis) and the payment calendar to the schedule's.
        self._schedule: Schedule = schedule
        self._index: ZeroInflationIndex = index
        self._base_cpi: float | None = base_cpi
        self._observation_lag: Period = observation_lag
        self._notionals: list[float] = []
        self._fixed_rates: list[float] = []
        self._payment_day_counter: DayCounter = Thirty360(Thirty360Convention.BondBasis)
        self._payment_adjustment: BusinessDayConvention = (
            BusinessDayConvention.ModifiedFollowing
        )
        self._payment_calendar: Calendar = schedule.calendar
        self._observation_interpolation: InterpolationType = InterpolationType.Flat
        self._subtract_inflation_nominal: bool = True
        self._caps: list[float] = []
        self._floors: list[float] = []
        self._ex_coupon_period: Period | None = None
        self._ex_coupon_calendar: Calendar | None = None
        self._ex_coupon_adjustment: BusinessDayConvention = BusinessDayConvention.Following
        self._ex_coupon_end_of_month: bool = False
        self._base_date: Date | None = None

    # --- chained setters -----------------------------------------------

    def with_notionals(self, notionals: float | Sequence[float]) -> CPILeg:
        """# C++ parity: ``withNotionals`` (.cpp:204-212)."""
        self._notionals = cfv.as_float_list(notionals)
        return self

    def with_fixed_rates(self, fixed_rates: float | Sequence[float]) -> CPILeg:
        """# C++ parity: ``withFixedRates`` (.cpp:194-202).

        A period whose fixed rate is exactly ``0.0`` degenerates to a
        :class:`~pquantlib.cashflows.fixed_rate_coupon.FixedRateCoupon`
        paying the cap/floor-clamped zero rate, as in C++.
        """
        self._fixed_rates = cfv.as_float_list(fixed_rates)
        return self

    def with_payment_day_counter(self, day_counter: DayCounter) -> CPILeg:
        """# C++ parity: ``withPaymentDayCounter`` (.cpp:220-223)."""
        self._payment_day_counter = day_counter
        return self

    def with_payment_adjustment(self, convention: BusinessDayConvention) -> CPILeg:
        """# C++ parity: ``withPaymentAdjustment`` (.cpp:225-228)."""
        self._payment_adjustment = convention
        return self

    def with_payment_calendar(self, calendar: Calendar) -> CPILeg:
        """# C++ parity: ``withPaymentCalendar`` (.cpp:230-233)."""
        self._payment_calendar = calendar
        return self

    def with_observation_interpolation(self, interpolation: InterpolationType) -> CPILeg:
        """# C++ parity: ``withObservationInterpolation`` (.cpp:188-191)."""
        self._observation_interpolation = interpolation
        return self

    def with_subtract_inflation_nominal(self, growth_only: bool) -> CPILeg:
        """# C++ parity: ``withSubtractInflationNominal`` (.cpp:214-218).

        Controls whether the final :class:`CPICashFlow` pays the inflated
        notional or only its growth.
        """
        self._subtract_inflation_nominal = growth_only
        return self

    def with_caps(self, caps: float | Sequence[float]) -> CPILeg:
        """# C++ parity: ``withCaps`` (.cpp:235-243).

        Only observable on zero-fixed-rate periods, which degenerate to a
        fixed coupon paying ``effectiveFixedRate({}, caps, floors, i)``. A
        cap on a non-zero-rate CPI coupon is a C++ ``QL_FAIL``.
        """
        self._caps = cfv.as_float_list(caps)
        return self

    def with_floors(self, floors: float | Sequence[float]) -> CPILeg:
        """# C++ parity: ``withFloors`` (.cpp:245-253) — see :meth:`with_caps`."""
        self._floors = cfv.as_float_list(floors)
        return self

    def with_ex_coupon_period(
        self,
        period: Period,
        calendar: Calendar,
        convention: BusinessDayConvention,
        end_of_month: bool = False,
    ) -> CPILeg:
        """# C++ parity: ``withExCouponPeriod`` (.cpp:255-266)."""
        self._ex_coupon_period = period
        self._ex_coupon_calendar = calendar
        self._ex_coupon_adjustment = convention
        self._ex_coupon_end_of_month = end_of_month
        return self

    def with_base_date(self, base_date: Date) -> CPILeg:
        """# C++ parity: ``withBaseDate`` (.cpp:268-271)."""
        self._base_date = base_date
        return self

    # --- operator Leg() -------------------------------------------------

    def build(self) -> list[CashFlow]:
        """Build the leg.

        # C++ parity: ``CPILeg::operator Leg()`` (.cpp:274-350).
        """
        # Local import for the coupon ↔ pricer cycle (see CPICoupon above).
        from pquantlib.cashflows.cpi_coupon_pricer import CPICouponPricer  # noqa: PLC0415
        from pquantlib.cashflows.inflation_coupon_pricer import (  # noqa: PLC0415
            set_coupon_pricer,
        )

        qassert.require(len(self._notionals) > 0, "no notional given")
        schedule = self._schedule
        n = len(schedule) - 1
        leg: list[CashFlow] = []

        base_date = self._base_date
        if n > 0:
            qassert.require(len(self._fixed_rates) > 0, "no fixedRates given")
            if self._base_date is None and self._base_cpi is None:
                base_date = schedule.date(0) - self._observation_lag

            for i in range(n):
                start = schedule.date(i)
                end = schedule.date(i + 1)
                payment_date = self._payment_calendar.adjust(end, self._payment_adjustment)
                ex_coupon_date: Date | None = None
                if self._ex_coupon_period is not None:
                    cal = self._ex_coupon_calendar
                    assert cal is not None
                    ex_coupon_date = cal.advance(
                        payment_date,
                        -self._ex_coupon_period.length,
                        self._ex_coupon_period.units,
                        self._ex_coupon_adjustment,
                        self._ex_coupon_end_of_month,
                    )
                ref_start, ref_end = start, end
                bdc = schedule.business_day_convention
                if schedule.has_is_regular() and schedule.has_tenor():
                    if i == 0 and not schedule.is_regular_at(1):
                        ref_start = schedule.calendar.adjust(end - schedule.tenor, bdc)
                    if i == n - 1 and not schedule.is_regular_at(i + 1):
                        ref_end = schedule.calendar.adjust(start + schedule.tenor, bdc)

                if cfv.get(self._fixed_rates, i, 1.0) == 0.0:
                    leg.append(
                        FixedRateCoupon.from_rate(
                            payment_date,
                            cfv.get(self._notionals, i, 0.0),
                            cfv.effective_fixed_rate([], self._caps, self._floors, i),
                            self._payment_day_counter,
                            start,
                            end,
                            ref_start,
                            ref_end,
                            ex_coupon_date,
                        )
                    )
                    continue
                qassert.require(
                    cfv.no_option(self._caps, self._floors, i),
                    "caps/floors on CPI coupons not implemented.",
                )
                leg.append(
                    CPICoupon(
                        payment_date,
                        cfv.get(self._notionals, i, 0.0),
                        start,
                        end,
                        self._index,
                        self._observation_lag,
                        self._observation_interpolation,
                        self._payment_day_counter,
                        cfv.get(self._fixed_rates, i, 0.0),
                        self._base_cpi,
                        base_date,
                        ref_start,
                        ref_end,
                        ex_coupon_date,
                    )
                )

        # in CPI legs you always have a notional flow of some sort
        payment_date = self._payment_calendar.adjust(
            schedule.date(n), self._payment_adjustment
        )
        leg.append(
            CPICashFlow(
                cfv.get(self._notionals, n, 0.0),
                self._index,
                base_date if base_date is not None else _NULL_DATE,
                self._base_cpi,
                schedule.date(n),
                self._observation_lag,
                self._observation_interpolation,
                payment_date,
                self._subtract_inflation_nominal,
            )
        )
        # no caps and floors here, so this is enough
        set_coupon_pricer(leg, CPICouponPricer())
        return leg


__all__ = ["CPICashFlow", "CPICoupon", "CPILeg"]
