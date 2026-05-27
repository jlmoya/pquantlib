"""FixedRateCoupon — coupon paying a fixed rate.

# C++ parity: ql/cashflows/fixedratecoupon.hpp + .cpp (v1.42.1).

amount = nominal * (rate.compound_factor(start, end, refStart, refEnd) - 1)

The C++ class has two ctors — one taking a raw Rate + DayCounter (which
internally builds an InterestRate(Simple, Annual)), and one taking a
fully-constructed ``InterestRate``. Python exposes a single ``__init__``
plus ``FixedRateCoupon.from_rate(...)`` classmethod for the raw-rate
variant.
"""

from __future__ import annotations

from pquantlib.cashflows.coupon import Coupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.interest_rate import InterestRate
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency


class FixedRateCoupon(Coupon):
    """Coupon paying a fixed interest rate.

    The interest rate carries the day counter + compounding + frequency,
    so the coupon's ``day_counter()`` is derived from the rate.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        interest_rate: InterestRate,
        accrual_start_date: Date,
        accrual_end_date: Date,
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
        self._rate: InterestRate = interest_rate

    @classmethod
    def from_rate(
        cls,
        payment_date: Date,
        nominal: float,
        rate: float,
        day_counter: DayCounter,
        accrual_start_date: Date,
        accrual_end_date: Date,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
        ex_coupon_date: Date | None = None,
    ) -> FixedRateCoupon:
        """C++ ctor 1: build a Simple/Annual InterestRate internally.

        C++ parity: ql/cashflows/fixedratecoupon.cpp:32-43.
        """
        ir = InterestRate(rate, day_counter, Compounding.Simple, Frequency.Annual)
        return cls(
            payment_date,
            nominal,
            ir,
            accrual_start_date,
            accrual_end_date,
            ref_period_start,
            ref_period_end,
            ex_coupon_date,
        )

    # --- Coupon interface ----------------------------------------------

    def rate(self) -> float:
        return self._rate.rate()

    def interest_rate(self) -> InterestRate:
        return self._rate

    def day_counter(self) -> DayCounter:
        return self._rate.day_counter()

    def amount(self) -> float:
        """amount = nominal * (compound_factor(start, end, ref_start, ref_end) - 1).

        C++ parity: ql/cashflows/fixedratecoupon.cpp:62-71.
        """
        return self._nominal * (
            self._rate.compound_factor_dates(
                self._accrual_start_date,
                self._accrual_end_date,
                self._ref_period_start,
                self._ref_period_end,
            )
            - 1.0
        )

    def accrued_amount(self, d: Date) -> float:
        """Accrued amount at the given date.

        C++ parity: ql/cashflows/fixedratecoupon.cpp:73-89.
        """
        if d <= self._accrual_start_date or d > self._payment_date:
            return 0.0
        if self.trading_ex_coupon(d):
            end = max(d, self._accrual_end_date)
            return -self._nominal * (
                self._rate.compound_factor_dates(
                    d, end, self._ref_period_start, self._ref_period_end
                )
                - 1.0
            )
        end = min(d, self._accrual_end_date)
        return self._nominal * (
            self._rate.compound_factor_dates(
                self._accrual_start_date,
                end,
                self._ref_period_start,
                self._ref_period_end,
            )
            - 1.0
        )
