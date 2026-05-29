"""IntegralCDOEngine — Riemann-integral CDO engine.

# C++ parity: ql/experimental/credit/integralcdoengine.{hpp,cpp} (v1.42.1).

Partitions each coupon's accrual period into ``step_size`` sub-intervals
and approximates the protection + premium legs as a Riemann sum on the
expected-tranche-loss differences.

For each sub-interval ``[d0, d]``:

- Premium leg: ``(tranche_notional - E_2) * running_rate * dt * D(d)``
  where ``dt`` is the day-count fraction ``d0 -> d`` and ``D(d)`` is the
  discount factor.
- Protection leg: ``(E_2 - E_1) * D(d)`` where ``E_i`` is the basket's
  expected tranche loss at ``d_i``.

The error counter increments whenever the integrated expected loss
decreases (a model artifact in C++; the Python port preserves the
diagnostic).

Side convention:
- Seller: receives premium, pays protection (sign as computed).
- Buyer: flips signs on protection / premium / upfront.
"""

from __future__ import annotations

from typing import cast

from pquantlib.cashflows.coupon import Coupon
from pquantlib.experimental.credit.synthetic_cdo import (
    ProtectionSide,
    SyntheticCDOArguments,
    SyntheticCDOResults,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def _months(n: int) -> Period:
    return Period(n, TimeUnit.Months)


class IntegralCDOEngine(
    GenericEngine[SyntheticCDOArguments, SyntheticCDOResults],
):
    """Riemann-integral CDO engine.

    Parameters
    ----------
    discount_curve
        Yield-term structure for premium-leg discounting.
    step_size
        Period sub-interval for Riemann partitioning (default: 3 Months,
        matching the C++ default).
    """

    def __init__(
        self,
        discount_curve: YieldTermStructure,
        step_size: Period | None = None,
    ) -> None:
        super().__init__(SyntheticCDOArguments(), SyntheticCDOResults())
        self._discount_curve: YieldTermStructure = discount_curve
        self._step_size: Period = step_size if step_size is not None else _months(3)
        discount_curve.register_with(self)

    def calculate(self) -> None:  # noqa: PLR0915
        """Compute Synthetic CDO NPV / premium / protection / upfront.

        # C++ parity: integralcdoengine.cpp:30-127.

        Loop nesting + Riemann inner step accounts for the high
        statement / branch counts.
        """
        args = self._arguments
        results = self._results
        assert args.basket is not None
        assert args.side is not None
        assert args.upfront_rate is not None
        assert args.running_rate is not None
        assert args.day_counter is not None

        today = ObservableSettings().evaluation_date_or_today()
        basket = args.basket

        results.protection_value = 0.0
        results.premium_value = 0.0
        results.upfront_premium_value = 0.0
        results.error = 0
        results.expected_tranche_loss = []
        results.x_min = basket.attachment_amount()
        results.x_max = basket.detachment_amount()
        results.remaining_notional = results.x_max - results.x_min
        inception_tranche_notional = basket.tranche_notional()

        null_calendar = NullCalendar()

        # E_1: expected tranche loss at the start of the first relevant period.
        e1: float = 0.0
        first_cf = args.normalized_leg[0]
        if not first_cf.has_occurred(today):
            first_coupon = cast("Coupon", first_cf)
            e1 = basket.expected_tranche_loss(first_coupon.accrual_start_date())
        results.expected_tranche_loss.append(e1)

        for cf in args.normalized_leg:
            if cf.has_occurred(today):
                results.expected_tranche_loss.append(0.0)
                continue

            coupon = cast("Coupon", cf)
            d1 = coupon.accrual_start_date()
            d2 = coupon.date()

            d0 = d1
            e2: float = 0.0
            # Riemann sub-partition by step_size from d0 -> d2.
            while True:
                start = d0 if d0 > today else today
                d = null_calendar.advance(
                    start,
                    self._step_size.length,
                    self._step_size.units,
                )
                d = min(d, d2)

                e2 = basket.expected_tranche_loss(d)

                results.premium_value += (
                    (inception_tranche_notional - e2)
                    * args.running_rate
                    * args.day_counter.year_fraction(d0, d)
                    * self._discount_curve.discount(d)
                )

                if e2 < e1:
                    results.error += 1

                results.protection_value += (
                    (e2 - e1) * self._discount_curve.discount(d)
                )

                d0 = d
                e1 = e2

                if d >= d2:
                    break
            results.expected_tranche_loss.append(e2)

        # Upfront — paid up-front at the first accrual start, discounted.
        if not args.normalized_leg[0].has_occurred(today):
            first_coupon = cast("Coupon", args.normalized_leg[0])
            results.upfront_premium_value = (
                inception_tranche_notional
                * args.upfront_rate
                * self._discount_curve.discount(first_coupon.accrual_start_date())
            )

        if args.side == ProtectionSide.Buyer:
            results.protection_value *= -1.0
            results.premium_value *= -1.0
            results.upfront_premium_value *= -1.0

        results.value = (
            results.premium_value
            - results.protection_value
            + results.upfront_premium_value
        )
        results.error_estimate = None

        # Fair spread GIVEN the upfront.
        fair_spread = 0.0
        if results.premium_value != 0.0:
            fair_spread = -(
                (results.protection_value + results.upfront_premium_value)
                * args.running_rate
                / results.premium_value
            )
        results.additional_results["fair_premium"] = fair_spread
        results.additional_results["premium_leg_npv"] = (
            results.premium_value + results.upfront_premium_value
        )
        results.additional_results["protection_leg_npv"] = results.protection_value


__all__ = ["IntegralCDOEngine"]
