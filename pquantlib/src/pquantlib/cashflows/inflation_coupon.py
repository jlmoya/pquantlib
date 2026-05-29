"""InflationCoupon — abstract coupon paying an inflation-index ratio.

# C++ parity: ql/cashflows/inflationcoupon.{hpp,cpp} (v1.42.1).

The C++ ``InflationCoupon`` extends ``Coupon`` with an inflation-index
slot, an ``observationLag`` period (mirroring the lag at which the
index publishes its fixings), and a ``fixingDays`` count. It defers the
rate to an ``InflationCouponPricer`` set via ``setPricer``; the pricer
must pass a ``checkPricerImpl`` predicate (Python: see concrete subclass
overrides).

Python divergences from C++:

- We narrow the index slot to ``InflationIndexProtocol`` to mirror the
  IborCoupon → IborIndexProtocol pattern from L2-D. Concrete subclasses
  (``CPICoupon``, ``YoYInflationCoupon``) further narrow at construction
  time via ``isinstance``/runtime checks.
- The C++ ``LazyObject::performCalculations`` deferred-execution cache
  is replaced by recomputing on every ``rate()`` call (matches the
  Phase-2 ``FloatingRateCoupon`` choice).
- ``Visitor`` accept() machinery is omitted (carve-out from L2-D).
- The C++ ``price(handle<YieldTermStructure>)`` helper is kept on the
  abstract for parity but uses the cross-cluster
  ``YieldTermStructureProtocol`` typing.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.coupon import Coupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.inflation.protocols import InflationIndexProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.cashflows.inflation_coupon_pricer import InflationCouponPricer
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


class InflationCoupon(Coupon):
    """Abstract coupon paying an inflation-index ratio.

    # C++ parity: ``InflationCoupon`` in ql/cashflows/inflationcoupon.hpp.
    Concrete subclasses (``CPICoupon``, ``YoYInflationCoupon``) override
    ``check_pricer_impl`` to enforce the pricer subtype contract.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        accrual_start_date: Date,
        accrual_end_date: Date,
        fixing_days: int,
        index: InflationIndexProtocol,
        observation_lag: Period,
        day_counter: DayCounter,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
        ex_coupon_date: Date | None = None,
    ) -> None:
        super().__init__(
            payment_date,
            nominal,
            accrual_start_date,
            accrual_end_date,
            ref_period_start,
            ref_period_end,
            ex_coupon_date,
        )
        self._index: InflationIndexProtocol = index
        self._observation_lag: Period = observation_lag
        self._day_counter: DayCounter = day_counter
        self._fixing_days: int = fixing_days
        self._pricer: InflationCouponPricer | None = None

    # ---- inspectors ---------------------------------------------------

    def index(self) -> InflationIndexProtocol:
        """C++ parity: ``InflationCoupon::index``."""
        return self._index

    def observation_lag(self) -> Period:
        """C++ parity: ``InflationCoupon::observationLag``."""
        return self._observation_lag

    def fixing_days(self) -> int:
        """C++ parity: ``InflationCoupon::fixingDays``."""
        return self._fixing_days

    def day_counter(self) -> DayCounter:
        """C++ parity: ``InflationCoupon::dayCounter`` (inline accessor)."""
        return self._day_counter

    def pricer(self) -> InflationCouponPricer | None:
        return self._pricer

    # ---- fixing date --------------------------------------------------

    def fixing_date(self) -> Date:
        """Compute the fixing date.

        # C++ parity: ql/cashflows/inflationcoupon.cpp:87-92 — uses
        # ``index.fixingCalendar().advance(refPeriodEnd - observationLag,
        # -fixingDays, Days, ModifiedPreceding)``. We accept the same
        # convention here. For inflation indexes, the fixing calendar is
        # always ``NullCalendar`` so ``advance`` reduces to a serial
        # subtraction.
        """
        ref_end_minus_lag = self._ref_period_end - self._observation_lag
        return self._index.fixing_calendar().advance(
            ref_end_minus_lag,
            -self._fixing_days,
            TimeUnit.Days,
            BusinessDayConvention.ModifiedPreceding,
            False,
        )

    def index_fixing(self) -> float:
        """Underlying-index fixing at ``fixing_date``.

        # C++ parity: ql/cashflows/inflationcoupon.cpp:100-102. Subclasses
        # ``CPICoupon`` and ``YoYInflationCoupon`` override this to use
        # the lagged-fixing math (``CPI::laggedFixing`` / ``laggedYoYRate``).
        """
        return self._index.fixing(self.fixing_date(), False)

    # ---- pricer wiring ------------------------------------------------

    def set_pricer(self, pricer: InflationCouponPricer | None) -> None:
        """Attach (or detach) an inflation-coupon pricer.

        # C++ parity: ql/cashflows/inflationcoupon.cpp:53-61. We validate
        # pricer-subtype compatibility via ``check_pricer_impl`` (abstract
        # in this class — see ``CPICoupon`` / ``YoYInflationCoupon``).
        """
        if pricer is not None:
            qassert.require(
                self.check_pricer_impl(pricer),
                "pricer given is wrong type",
            )
        self._pricer = pricer

    @abstractmethod
    def check_pricer_impl(self, pricer: InflationCouponPricer) -> bool:
        """Concrete-class predicate: ``True`` iff ``pricer`` is the right subtype.

        # C++ parity: ``InflationCoupon::checkPricerImpl`` (pure virtual).
        """

    # ---- Coupon interface ---------------------------------------------

    def rate(self) -> float:
        """Inflation rate, as set by the pricer's ``swaplet_rate``.

        # C++ parity: ql/cashflows/inflationcoupon.cpp:64-66. The C++
        # version routes through ``LazyObject::calculate`` and caches the
        # rate; we recompute every call (consistent with our
        # ``FloatingRateCoupon`` choice).
        """
        qassert.require(self._pricer is not None, "pricer not set")
        assert self._pricer is not None
        self._pricer.initialize(self)
        return self._pricer.swaplet_rate()

    def amount(self) -> float:
        """C++ parity: ``InflationCoupon::amount`` inline at hpp:62 ==
        ``rate() * accrualPeriod() * nominal()``.
        """
        return self.rate() * self.accrual_period() * self._nominal

    def accrued_amount(self, d: Date) -> float:
        """C++ parity: ql/cashflows/inflationcoupon.cpp:78-84."""
        if d <= self._accrual_start_date or d > self._payment_date:
            return 0.0
        return self._nominal * self.rate() * self.accrued_period(d)

    def price(self, discounting_curve: YieldTermStructureProtocol) -> float:
        """Discounted amount on the payment date.

        # C++ parity: ql/cashflows/inflationcoupon.cpp:95-97.
        """
        return self.amount() * discounting_curve.discount(self.date())
