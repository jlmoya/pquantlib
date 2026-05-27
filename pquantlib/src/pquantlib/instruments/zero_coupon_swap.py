"""ZeroCouponSwap — single-payment-at-maturity fixed-vs-float swap.

# C++ parity: ql/instruments/zerocouponswap.{hpp,cpp} (v1.42.1).

The contract pays a *single* fixed cashflow ``N^FIX = N * [(1+R)^T - 1]``
and a *single* floating cashflow that equals
``N * [prod_k (1 + tau_k * L_k) - 1]`` where ``L_k`` are IBOR fixings
across the accrual sub-periods.

The C++ class builds the floating side via ``MultipleResetsCoupon`` with
the ``CompoundingMultipleResetsPricer``. Neither is yet ported (deferred
to L3-D), so PQuantLib's port uses a custom ``CompoundedIborCashFlow``
that walks the sub-period schedule with the index's forecast curve and
computes the compound floating amount directly. This is sufficient for
DiscountingSwapEngine pricing (the engine only reads ``.date()`` and
``.amount()`` from each cashflow).

C++ carve-outs deferred here:
- ``MultipleResetsCoupon`` proper (full coupon-with-pricer with
  separate fixing-end-dates etc.) — deferred to L3-D.
- ``averagingMethod = Simple`` (we always compound).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.simple_cash_flow import SimpleCashFlow
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.swap import Swap, SwapType
from pquantlib.interest_rate import InterestRate
from pquantlib.termstructures.protocols import IborIndexProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


class _CompoundedIborCashFlow(CashFlow):
    """Single cashflow whose amount is the IBOR-compounded notional growth.

    Approximates C++ ``MultipleResetsCoupon`` + ``CompoundingMultipleResetsPricer``
    just enough for DiscountingSwapEngine pricing of a ZeroCouponSwap.

    amount = nominal * (prod_k (1 + tau_k * L_k) - 1)
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        accrual_start: Date,
        accrual_end: Date,
        index: IborIndexProtocol,
    ) -> None:
        super().__init__()
        self._payment_date: Date = payment_date
        self._nominal: float = nominal
        self._index: IborIndexProtocol = index
        # Build sub-period schedule from start to maturity at the index
        # tenor (backward generation, like C++).
        self._schedule: Schedule = Schedule.from_rule(
            accrual_start,
            accrual_end,
            index.tenor(),
            index.fixing_calendar(),
            index.business_day_convention(),
            index.business_day_convention(),
            DateGeneration.Backward,
            index.end_of_month(),
        )

    def date(self) -> Date:
        return self._payment_date

    def amount(self) -> float:
        compound = 1.0
        cal = self._index.fixing_calendar()
        # Walk consecutive schedule pairs, fetching each fixing.
        for i in range(len(self._schedule) - 1):
            sub_start = self._schedule.date(i)
            sub_end = self._schedule.date(i + 1)
            tau = self._index.day_counter().year_fraction(sub_start, sub_end)
            # Fixing is at fixing_days before the period start.
            fix_date = cal.advance(
                sub_start,
                -self._index.fixing_days(),
                TimeUnit.Days,
                BusinessDayConvention.Preceding,
                False,
            )
            rate = self._index.fixing(fix_date, True)
            compound *= 1.0 + tau * rate
        return self._nominal * (compound - 1.0)


class ZeroCouponSwap(Swap):
    """Zero-coupon swap: single fixed cashflow vs single compounded-IBOR cashflow.

    # C++ parity: ``ZeroCouponSwap`` (zerocouponswap.{hpp,cpp}).
    """

    def __init__(
        self,
        swap_type: SwapType,
        base_nominal: float,
        start_date: Date,
        maturity_date: Date,
        fixed_payment_or_rate: float,
        ibor_index: IborIndexProtocol,
        payment_calendar: Calendar,
        payment_convention: BusinessDayConvention = BusinessDayConvention.Following,
        payment_delay: int = 0,
        fixed_day_counter: DayCounter | None = None,
    ) -> None:
        """Construct from either a known fixed payment amount or a fixed rate.

        If ``fixed_day_counter`` is provided, ``fixed_payment_or_rate`` is
        interpreted as a *rate* (the C++ ctor with explicit rate +
        day-counter). Otherwise it's interpreted as a known fixed *amount*
        (the C++ ctor with ``fixedPayment``).
        """
        super().__init__(n_legs=2)
        qassert.require(base_nominal >= 0.0, "base nominal cannot be negative")
        qassert.require(
            start_date < maturity_date,
            f"start date ({start_date}) later than or equal to maturity date ({maturity_date})",
        )

        self._swap_type: SwapType = swap_type
        self._base_nominal: float = base_nominal
        self._ibor_index: IborIndexProtocol = ibor_index
        self._start_date: Date = start_date
        self._maturity_date: Date = maturity_date
        self._payment_date: Date = payment_calendar.advance(
            maturity_date, payment_delay, TimeUnit.Days, payment_convention, False
        )

        # Floating leg: a single compounded-IBOR cashflow.
        floating_cf = _CompoundedIborCashFlow(
            self._payment_date,
            base_nominal,
            start_date,
            maturity_date,
            ibor_index,
        )
        self._legs[1] = [floating_cf]
        floating_cf.register_with(self)

        # Fixed leg: either a SimpleCashFlow (known amount) or a single
        # FixedRateCoupon (rate + day-counter).
        fixed_cf: CashFlow
        if fixed_day_counter is None:
            fixed_cf = SimpleCashFlow(fixed_payment_or_rate, self._payment_date)
        else:
            rate = InterestRate(
                fixed_payment_or_rate,
                fixed_day_counter,
                Compounding.Compounded,
                Frequency.Annual,
            )
            fixed_cf = FixedRateCoupon(
                self._payment_date,
                base_nominal,
                rate,
                start_date,
                maturity_date,
            )
        self._legs[0] = [fixed_cf]
        fixed_cf.register_with(self)

        # Payer sign: C++ Payer means "pay fixed". For ZeroCouponSwap the
        # ctor docs say "payer/receiver refers to the fixed leg" — so
        # Payer => fixed paid (sign -1), float received (+1).
        if swap_type == SwapType.Payer:
            self._payer[0] = -1.0
            self._payer[1] = +1.0
        elif swap_type == SwapType.Receiver:
            self._payer[0] = +1.0
            self._payer[1] = -1.0
        else:
            qassert.fail("unknown zero coupon swap type")

    # --- inspectors ---------------------------------------------------------

    def swap_type(self) -> SwapType:
        return self._swap_type

    def base_nominal(self) -> float:
        return self._base_nominal

    def start_date(self) -> Date:
        return self._start_date

    def maturity_date(self) -> Date:
        return self._maturity_date

    def ibor_index(self) -> IborIndexProtocol:
        return self._ibor_index

    def fixed_leg(self):  # type: ignore[no-untyped-def]
        return self._legs[0]

    def floating_leg(self):  # type: ignore[no-untyped-def]
        return self._legs[1]

    def fixed_payment(self) -> float:
        return self._legs[0][0].amount()

    # --- results ----------------------------------------------------------

    def fixed_leg_npv(self) -> float:
        return self.leg_npv(0)

    def floating_leg_npv(self) -> float:
        return self.leg_npv(1)

    def fair_fixed_payment(self) -> float:
        """Fair fixed payment so that NPV is zero.

        # C++ parity: ``ZeroCouponSwap::fairFixedPayment`` (zerocouponswap.cpp:145-154).
        """
        scaling = -1.0 if self.payer(1) else 1.0
        return self.floating_leg_npv() / (self.end_discounts(0) * scaling)

    def fair_fixed_rate(self, day_counter: DayCounter) -> float:
        """Implied compounded annual rate that makes the swap fair.

        # C++ parity: ``ZeroCouponSwap::fairFixedRate`` (zerocouponswap.cpp:156-165).
        """
        compound = self.fair_fixed_payment() / self._base_nominal + 1.0
        ir = InterestRate.implied_rate_dates(
            compound,
            day_counter,
            Compounding.Compounded,
            Frequency.Annual,
            self._start_date,
            self._maturity_date,
        )
        return ir.rate()


__all__ = ["ZeroCouponSwap"]
