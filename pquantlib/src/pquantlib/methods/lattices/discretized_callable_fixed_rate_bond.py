"""DiscretizedCallableFixedRateBond - callable bond on a tree lattice.

# C++ parity: ql/experimental/callablebonds/discretizedcallablefixedratebond.{hpp,cpp}
#             (v1.42.1).

Reflects a ``CallableBond``'s arguments onto a short-rate lattice so the
``TreeCallableFixedRateBondEngine`` can roll back the embedded American/
Bermudan call/put condition.

Algorithm (matches the C++ source exactly):

- ``reset`` seeds every node with the redemption (face x redemption/100).
- coupons land in ``post_adjust_values`` (default) - i.e. a coupon paid
  at a node is added *after* any callability applied at that node - unless
  a callability time was snapped onto a coupon time, in which case the
  coupon flips to ``pre`` adjustment (added before the callability, since
  from the rollback's perspective the later-in-time coupon must be applied
  first).
- callabilities land in ``post_adjust_values``: a *call* caps node values
  at the (dirty) call price; a *put* floors them at the put price.

Exercise-date snapping: if a callability time falls within one week
*before* a coupon date, it is snapped onto the coupon time and the call
price is rescaled by the ratio of spread-adjusted discount factors
between the (true) call date and the coupon date - preserving present
value across the snap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.instruments.callability import CallabilityType
from pquantlib.methods.lattices.discretized_asset import (
    CouponAdjustment,
    DiscretizedAsset,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

if TYPE_CHECKING:
    from pquantlib.instruments.callable_bond import CallableBondArguments
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure
    from pquantlib.time.date import Date

# One week as a fraction of a year (C++ ``dt = 1.0/52``).
_ONE_WEEK: float = 1.0 / 52


def _within_next_week(t1: float, t2: float) -> bool:
    # C++ parity: discretizedcallablefixedratebond.cpp:27-30.
    return t1 <= t2 <= t1 + _ONE_WEEK


class DiscretizedCallableFixedRateBond(DiscretizedAsset):
    """Lattice-discretized callable fixed-rate bond.

    # C++ parity: ``class DiscretizedCallableFixedRateBond``
    # (discretizedcallablefixedratebond.{hpp,cpp}, v1.42.1).
    """

    def __init__(
        self,
        args: CallableBondArguments,
        term_structure: YieldTermStructure,
    ) -> None:
        super().__init__()
        self._arguments: CallableBondArguments = args
        # Copy callability prices - they get rescaled by the snapping logic.
        self._adjusted_callability_prices: list[float] = list(args.callability_prices)

        day_counter = term_structure.day_counter()
        reference_date = term_structure.reference_date()

        self._redemption_time: float = day_counter.year_fraction(
            reference_date, args.redemption_date
        )

        # By default coupons adjust in postAdjustValuesImpl.
        self._coupon_adjustments: list[CouponAdjustment] = [
            CouponAdjustment.Post for _ in args.coupon_dates
        ]

        self._coupon_times: list[float] = [
            day_counter.year_fraction(reference_date, d) for d in args.coupon_dates
        ]

        self._callability_times: list[float] = [0.0] * len(args.callability_dates)

        spread = args.spread

        def df_incl_spread(date: Date) -> float:
            time = term_structure.time_from_reference(date)
            zero_incl = (
                term_structure.zero_rate(
                    date,
                    Compounding.Continuous,
                    Frequency.NoFrequency,
                    result_day_counter=term_structure.day_counter(),
                ).rate()
                + spread
            )
            return float(np.exp(-zero_incl * time))

        for i, callability_date in enumerate(args.callability_dates):
            callability_time = day_counter.year_fraction(reference_date, callability_date)

            # Snap exercise dates to the closest coupon date.
            for j, coupon_time in enumerate(self._coupon_times):
                coupon_date = args.coupon_dates[j]
                if _within_next_week(callability_time, coupon_time) and callability_date < coupon_date:
                    callability_time = coupon_time
                    # Order of events flips: coupon added pre (before callability).
                    self._coupon_adjustments[j] = CouponAdjustment.Pre
                    # Account for the missing discount factor (incl. any spread).
                    df_till_call = df_incl_spread(callability_date)
                    df_till_coupon = df_incl_spread(coupon_date)
                    self._adjusted_callability_prices[i] *= df_till_call / df_till_coupon
                    break

            self._adjusted_callability_prices[i] *= args.face_amount / 100.0
            self._callability_times[i] = callability_time

    # ------------------------------------------------------------------
    # DiscretizedAsset interface
    # ------------------------------------------------------------------

    def reset(self, size: int) -> None:
        # C++ parity: discretizedcallablefixedratebond.cpp:103-106.
        self._values = np.full(size, self._arguments.redemption, dtype=np.float64)
        self.adjust_values()

    def mandatory_times(self) -> list[float]:
        # C++ parity: discretizedcallablefixedratebond.cpp:109-134.
        times: list[float] = []
        if self._redemption_time >= 0.0:
            times.append(self._redemption_time)
        for t in self._coupon_times:
            if t >= 0.0:
                times.append(t)
        for t in self._callability_times:
            if t >= 0.0:
                times.append(t)
        return times

    def _pre_adjust_values_impl(self) -> None:
        # C++ parity: discretizedcallablefixedratebond.cpp:137-146.
        for i, t in enumerate(self._coupon_times):
            if self._coupon_adjustments[i] == CouponAdjustment.Pre and t >= 0.0 and self.is_on_time(t):
                self._add_coupon(i)

    def _post_adjust_values_impl(self) -> None:
        # C++ parity: discretizedcallablefixedratebond.cpp:149-165.
        for i, t in enumerate(self._callability_times):
            if t >= 0.0 and self.is_on_time(t):
                self._apply_callability(i)
        for i, t in enumerate(self._coupon_times):
            if self._coupon_adjustments[i] == CouponAdjustment.Post and t >= 0.0 and self.is_on_time(t):
                self._add_coupon(i)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_callability(self, i: int) -> None:
        # C++ parity: discretizedcallablefixedratebond.cpp:168-184.
        price = self._adjusted_callability_prices[i]
        c_type = self._arguments.put_call_schedule[i].type()
        if c_type == CallabilityType.Call:
            self._values = np.minimum(price, self._values)
        elif c_type == CallabilityType.Put:
            self._values = np.maximum(self._values, price)
        else:  # pragma: no cover - exhaustive
            qassert.fail("unknown callability type")

    def _add_coupon(self, i: int) -> None:
        # C++ parity: discretizedcallablefixedratebond.cpp:187-189.
        self._values = self._values + self._arguments.coupon_amounts[i]


__all__ = ["DiscretizedCallableFixedRateBond"]
