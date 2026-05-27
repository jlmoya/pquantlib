"""Coupon — abstract coupon accruing over a fixed period.

# C++ parity: ql/cashflows/coupon.hpp + ql/cashflows/coupon.cpp (v1.42.1).

The C++ ``Coupon`` adds nominal, accrual_start_date, accrual_end_date,
ref_period_start, ref_period_end, ex_coupon_date and provides
``accrualPeriod()`` / ``accrualDays()`` / ``accruedPeriod(d)`` /
``accruedDays(d)`` derived from ``dayCounter()``. ``rate()``,
``dayCounter()``, and ``accruedAmount(d)`` are abstract.

Python design notes:
- ``rate``, ``day_counter``, and ``accrued_amount`` are abstract methods.
- ``date()`` is concretized to return ``payment_date_``.
- ``accrual_period`` is computed eagerly on first call and memoized
  (mirrors the C++ ``mutable Real accrualPeriod_`` cache).
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.date import Date

# Module-level null Date constant for default args (avoids B008).
_NULL_DATE: Date = Date()


class Coupon(CashFlow):
    """Abstract coupon over an accrual period.

    Constructor mirrors C++ ``Coupon::Coupon`` (ql/cashflows/coupon.cpp:27-42):
    if ``ref_period_start`` is null, fall back to ``accrual_start_date``;
    if ``ref_period_end`` is null, fall back to ``accrual_end_date``.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        accrual_start_date: Date,
        accrual_end_date: Date,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
        ex_coupon_date: Date | None = None,
    ) -> None:
        super().__init__()
        self._payment_date: Date = payment_date
        self._nominal: float = nominal
        self._accrual_start_date: Date = accrual_start_date
        self._accrual_end_date: Date = accrual_end_date
        self._ref_period_start: Date = (
            ref_period_start if ref_period_start is not None else accrual_start_date
        )
        self._ref_period_end: Date = (
            ref_period_end if ref_period_end is not None else accrual_end_date
        )
        self._ex_coupon_date: Date = ex_coupon_date if ex_coupon_date is not None else _NULL_DATE
        self._accrual_period_cache: float | None = None

    # --- CashFlow interface --------------------------------------------

    def date(self) -> Date:
        return self._payment_date

    def ex_coupon_date(self) -> Date:
        return self._ex_coupon_date

    # --- inspectors ----------------------------------------------------

    def nominal(self) -> float:
        return self._nominal

    def accrual_start_date(self) -> Date:
        return self._accrual_start_date

    def accrual_end_date(self) -> Date:
        return self._accrual_end_date

    def reference_period_start(self) -> Date:
        return self._ref_period_start

    def reference_period_end(self) -> Date:
        return self._ref_period_end

    def accrual_period(self) -> float:
        """Accrual period as a year fraction.

        C++ parity: ql/cashflows/coupon.cpp:44-50 — cached lazily.
        """
        if self._accrual_period_cache is None:
            self._accrual_period_cache = self.day_counter().year_fraction(
                self._accrual_start_date,
                self._accrual_end_date,
                self._ref_period_start,
                self._ref_period_end,
            )
        return self._accrual_period_cache

    def accrual_days(self) -> int:
        """Accrual period in days.

        C++ parity: ql/cashflows/coupon.cpp:52-55.
        """
        return self.day_counter().day_count(self._accrual_start_date, self._accrual_end_date)

    def accrued_period(self, d: Date) -> float:
        """Accrued period as fraction of year at the given date.

        C++ parity: ql/cashflows/coupon.cpp:57-69.
        """
        if d <= self._accrual_start_date or d > self._payment_date:
            return 0.0
        if self.trading_ex_coupon(d):
            end = max(d, self._accrual_end_date)
            return -self.day_counter().year_fraction(
                d, end, self._ref_period_start, self._ref_period_end
            )
        end = min(d, self._accrual_end_date)
        return self.day_counter().year_fraction(
            self._accrual_start_date,
            end,
            self._ref_period_start,
            self._ref_period_end,
        )

    def accrued_days(self, d: Date) -> int:
        """Accrued days at the given date.

        C++ parity: ql/cashflows/coupon.cpp:71-78.
        """
        if d <= self._accrual_start_date or d > self._payment_date:
            return 0
        end = min(d, self._accrual_end_date)
        return self.day_counter().day_count(self._accrual_start_date, end)

    # --- abstract Coupon-interface members -----------------------------

    @abstractmethod
    def rate(self) -> float:
        """Accrued rate."""

    @abstractmethod
    def day_counter(self) -> DayCounter:
        """Day counter for accrual calculation."""

    @abstractmethod
    def accrued_amount(self, d: Date) -> float:
        """Accrued amount at the given date."""
