"""PricingPeriod — a pricing period in an energy swap.

# C++ parity: ql/experimental/commodities/pricingperiod.hpp (v1.42.1).

A :class:`DateInterval` augmented with a payment date and the quantity
priced over the period.

# C++ parity note: the v1.42.1 ``PricingPeriod`` ctor takes
# ``(startDate, endDate, paymentDate, quantity)`` — there is *no* unit-cost
# argument in this version (an earlier QuantLib carried one). We follow the
# pinned v1.42.1 header. The ``PricingPeriods`` typedef
# (``vector<shared_ptr<PricingPeriod>>``) maps to ``list[PricingPeriod]``.
"""

from __future__ import annotations

from pquantlib.experimental.commodities.date_interval import DateInterval
from pquantlib.experimental.commodities.quantity import Quantity
from pquantlib.time.date import Date


class PricingPeriod(DateInterval):
    """A pricing period: a date interval + payment date + priced quantity."""

    def __init__(
        self,
        start_date: Date,
        end_date: Date,
        payment_date: Date,
        quantity: Quantity,
    ) -> None:
        super().__init__(start_date, end_date)
        self._payment_date: Date = payment_date
        self._quantity: Quantity = quantity

    @property
    def payment_date(self) -> Date:
        return self._payment_date

    @property
    def quantity(self) -> Quantity:
        return self._quantity

    def __repr__(self) -> str:
        return (
            f"PricingPeriod({self.start_date} to {self.end_date}, "
            f"pay {self._payment_date})"
        )


# C++ parity: ``typedef std::vector<ext::shared_ptr<PricingPeriod>> PricingPeriods;``
PricingPeriods = list[PricingPeriod]
