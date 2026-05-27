"""ZeroCouponBond — single-redemption no-coupon bond.

# C++ parity: ql/instruments/bonds/zerocouponbond.{hpp,cpp} (v1.42.1).
"""

from __future__ import annotations

from pquantlib.instruments.bond import Bond
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class ZeroCouponBond(Bond):
    """Zero-coupon bond.

    # C++ parity: zerocouponbond.cpp:26-39.
    """

    def __init__(
        self,
        settlement_days: int,
        calendar: Calendar,
        face_amount: float,
        maturity_date: Date,
        payment_convention: BusinessDayConvention = BusinessDayConvention.Following,
        redemption: float = 100.0,
        issue_date: Date | None = None,
    ) -> None:
        Bond.__init__(self, settlement_days, calendar, issue_date)
        self._maturity_date = maturity_date
        redemption_date = calendar.adjust(maturity_date, payment_convention)
        self._set_single_redemption(face_amount, redemption, redemption_date)
        for cf in self._cashflows:
            cf.register_with(self)


__all__ = ["ZeroCouponBond"]
