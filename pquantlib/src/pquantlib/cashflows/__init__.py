"""Cash flow primitives.

# C++ parity: ql/cashflow.hpp + ql/cashflows/*.hpp (v1.42.1).

Public surface in alphabetical order:

- ``CashFlow`` (abstract base, observable)
- ``CashFlows`` (free-function aggregator namespace via class with static methods)
- ``Compounding`` (IntEnum: Simple/Compounded/Continuous/+2 hybrid)
- ``Coupon`` (abstract, accrual-period machinery)
- ``CouponPricer`` (abstract base for floating-rate coupon pricers)
- ``Duration`` (IntEnum: Simple/Macaulay/Modified)
- ``FixedRateCoupon`` (concrete fixed-rate coupon)
- ``FloatingRateCoupon`` (abstract floating-rate coupon)
- ``IborCoupon`` (concrete IBOR-indexed coupon)
- ``IborCouponPricer`` (concrete trivial pricer for plain IBOR coupons)
- ``BlackIborCouponPricer`` (Black-vol pricer — without cap/floor support)
- ``InterestRate`` (frozen dataclass — rate + day_counter + compounding + frequency)
- ``OvernightIndexedCoupon`` (concrete OIS-style daily-compounded coupon)
- ``SimpleCashFlow`` (fixed amount on a date)

Free functions:

- ``fixed_rate_leg``
- ``ibor_leg``
- ``overnight_leg``
"""

from __future__ import annotations
