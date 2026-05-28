"""DiscretizedSwap — vanilla swap reflected on a tree/FD lattice.

# C++ parity: ql/pricingengines/swap/discretizedswap.{hpp,cpp} (v1.42.1).

Discretizes a ``VanillaSwap`` (a.k.a. ``FixedVsFloatingSwap``) on the
lattice. At each fixed/floating reset time the discretization adds
the period's contribution via a temporary ``DiscretizedDiscountBond``
rolled back from the matching pay time.

The C++ class is logically owned by ``ql/pricingengines/swap/`` but
its functional surface is shared with the
``TreeSwaptionEngine``/``TreeCapFloorEngine`` lattice machinery, so
PQuantLib parks it under ``methods/lattices`` alongside its peers.

Per-coupon ``CouponAdjustment`` (pre / post) controls *when* the
coupon contribution lands during the rollback adjustment:

  * ``pre``  — added by ``preAdjustValuesImpl`` (before the option's
    exercise condition, i.e. the swap value at the reset time
    *excludes* the coupon).
  * ``post`` — added by ``postAdjustValuesImpl`` (the swap value at
    the reset time *includes* the coupon).

The swap argument carrier (``FixedVsFloatingSwapArguments``) carries
``fixed_reset_dates`` / ``fixed_pay_dates`` / ``floating_reset_dates``
/ ``floating_pay_dates`` plus the per-coupon ``fixed_coupons`` /
``floating_coupons`` / ``floating_accrual_times`` / ``floating_spreads``
arrays used in the per-period BPS calculation.

Past-fixings handling: if a reset is strictly before reference_date
but the pay is in the future (or the include-todays toggle is on),
the C++ algorithm flips the coupon to ``post`` adjustment and uses
the *cached* coupon amount (``args.floating_coupons[i]`` already
populated by the calling instrument). PQuantLib carries the same
include-todays toggle off the global Settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.instruments.swap import SwapType
from pquantlib.methods.lattices.discretized_asset import (
    CouponAdjustment,
    DiscretizedAsset,
)
from pquantlib.methods.lattices.discretized_discount_bond import (
    DiscretizedDiscountBond,
)
from pquantlib.time.date import Date as _Date

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.instruments.fixed_vs_floating_swap import (
        FixedVsFloatingSwapArguments,
    )
    from pquantlib.time.date import Date


class DiscretizedSwap(DiscretizedAsset):
    """Discretized vanilla swap on the lattice.

    # C++ parity: ``class DiscretizedSwap`` (discretizedswap.{hpp,cpp}, v1.42.1).
    """

    def __init__(
        self,
        args: FixedVsFloatingSwapArguments,
        reference_date: Date,
        day_counter: DayCounter,
        fixed_coupon_adjustments: list[CouponAdjustment] | None = None,
        floating_coupon_adjustments: list[CouponAdjustment] | None = None,
    ) -> None:
        # # C++ parity: discretizedswap.cpp:36-95 (the two ctor overloads;
        # # the single-arg overload defaults adjustments to all-pre).
        super().__init__()
        self._arguments: FixedVsFloatingSwapArguments = args
        n_fixed = len(args.fixed_pay_dates)
        n_float = len(args.floating_pay_dates)

        if fixed_coupon_adjustments is None:
            fixed_coupon_adjustments = [CouponAdjustment.Pre] * n_fixed
        if floating_coupon_adjustments is None:
            floating_coupon_adjustments = [CouponAdjustment.Pre] * n_float

        qassert.require(
            len(fixed_coupon_adjustments) == n_fixed,
            "fixed coupon adjustments size mismatch",
        )
        qassert.require(
            len(floating_coupon_adjustments) == n_float,
            "floating coupon adjustments size mismatch",
        )

        # Settings toggle for include-todays-cashflows.
        # # C++ parity: discretizedswap.cpp:61-63 — reads from
        # Settings::instance().includeTodaysCashFlows() (an optional<bool>).
        # PQuantLib's Settings doesn't separately track this yet; default
        # to False (matches the C++ default after the optional unwrap).
        include_todays = False

        # Year-fraction tables for the fixed leg.
        self._fixed_reset_times: list[float] = []
        self._fixed_pay_times: list[float] = []
        self._fixed_reset_in_past: list[bool] = []
        self._fixed_coupon_adjustments = list(fixed_coupon_adjustments)
        for i in range(n_fixed):
            reset_d = args.fixed_reset_dates[i]
            pay_d = args.fixed_pay_dates[i]
            assert isinstance(reset_d, _Date)
            assert isinstance(pay_d, _Date)
            rt = day_counter.year_fraction(reference_date, reset_d)
            pt = day_counter.year_fraction(reference_date, pay_d)
            in_past = self._is_reset_in_past(rt, pt, include_todays)
            self._fixed_reset_times.append(rt)
            self._fixed_pay_times.append(pt)
            self._fixed_reset_in_past.append(in_past)
            if in_past:
                # # C++ parity: discretizedswap.cpp:75-76.
                self._fixed_coupon_adjustments[i] = CouponAdjustment.Post

        # Year-fraction tables for the floating leg.
        self._floating_reset_times: list[float] = []
        self._floating_pay_times: list[float] = []
        self._floating_reset_in_past: list[bool] = []
        self._floating_coupon_adjustments = list(floating_coupon_adjustments)
        for i in range(n_float):
            reset_d = args.floating_reset_dates[i]
            pay_d = args.floating_pay_dates[i]
            assert isinstance(reset_d, _Date)
            assert isinstance(pay_d, _Date)
            rt = day_counter.year_fraction(reference_date, reset_d)
            pt = day_counter.year_fraction(reference_date, pay_d)
            in_past = self._is_reset_in_past(rt, pt, include_todays)
            self._floating_reset_times.append(rt)
            self._floating_pay_times.append(pt)
            self._floating_reset_in_past.append(in_past)
            if in_past:
                self._floating_coupon_adjustments[i] = CouponAdjustment.Post

    @staticmethod
    def _is_reset_in_past(reset_time: float, pay_time: float, include_todays: bool) -> bool:
        # # C++ parity: discretizedswap.cpp:28-33 — the anonymous-ns
        # ``isResetTimeInPast`` helper.
        return reset_time < 0.0 and (pay_time > 0.0 or (include_todays and pay_time == 0.0))

    # --- DiscretizedAsset interface --------------------------------------

    def reset(self, size: int) -> None:
        """Zero-init the values array and run the initial adjustment.

        # C++ parity: ``DiscretizedSwap::reset`` (discretizedswap.cpp:97-100).
        """
        self._values = np.zeros(size, dtype=np.float64)
        self.adjust_values()

    def mandatory_times(self) -> list[float]:
        """Union of non-negative reset + pay times on both legs.

        # C++ parity: ``DiscretizedSwap::mandatoryTimes`` (discretizedswap.cpp:102-121).
        """
        times: list[float] = []
        for t in (
            *self._fixed_reset_times,
            *self._fixed_pay_times,
            *self._floating_reset_times,
            *self._floating_pay_times,
        ):
            if t >= 0.0:
                times.append(t)
        return times

    # --- adjustments ------------------------------------------------------

    def _pre_adjust_values_impl(self) -> None:
        """Add pre-adjustment-marked coupons (floating first, then fixed).

        # C++ parity: ``DiscretizedSwap::preAdjustValuesImpl`` (discretizedswap.cpp:123-138).
        """
        # Floating leg.
        for i, t in enumerate(self._floating_reset_times):
            if (
                self._floating_coupon_adjustments[i] == CouponAdjustment.Pre
                and t >= 0.0
                and self.is_on_time(t)
            ):
                self._add_floating_coupon(i)
        # Fixed leg.
        for i, t in enumerate(self._fixed_reset_times):
            if (
                self._fixed_coupon_adjustments[i] == CouponAdjustment.Pre
                and t >= 0.0
                and self.is_on_time(t)
            ):
                self._add_fixed_coupon(i)

    def _post_adjust_values_impl(self) -> None:
        """Add post-adjustment coupons + past-reset cached coupons.

        # C++ parity: ``DiscretizedSwap::postAdjustValuesImpl`` (discretizedswap.cpp:140-182).
        """
        # Floating leg — post-adjusted.
        for i, t in enumerate(self._floating_reset_times):
            if (
                self._floating_coupon_adjustments[i] == CouponAdjustment.Post
                and t >= 0.0
                and self.is_on_time(t)
            ):
                self._add_floating_coupon(i)
        # Fixed leg — post-adjusted.
        for i, t in enumerate(self._fixed_reset_times):
            if (
                self._fixed_coupon_adjustments[i] == CouponAdjustment.Post
                and t >= 0.0
                and self.is_on_time(t)
            ):
                self._add_fixed_coupon(i)

        # Fixed coupons whose reset is in the past, at the pay time.
        for i, t in enumerate(self._fixed_pay_times):
            if self._fixed_reset_in_past[i] and self.is_on_time(t):
                fixed_coupon = self._arguments.fixed_coupons[i]
                if self._arguments.swap_type == SwapType.Payer:
                    self._values = self._values - fixed_coupon
                else:
                    self._values = self._values + fixed_coupon

        # Floating coupons whose reset is in the past, at the pay time.
        for i, t in enumerate(self._floating_pay_times):
            if self._floating_reset_in_past[i] and self.is_on_time(t):
                current_floating = self._arguments.floating_coupons[i]
                qassert.require(
                    current_floating is not None,
                    "current floating coupon not given",
                )
                assert current_floating is not None
                if self._arguments.swap_type == SwapType.Payer:
                    self._values = self._values + current_floating
                else:
                    self._values = self._values - current_floating

    # --- per-coupon contributions ----------------------------------------

    def _add_fixed_coupon(self, i: int) -> None:
        """Discount the i-th fixed coupon back to ``self.time`` and add.

        # C++ parity: ``DiscretizedSwap::addFixedCoupon`` (discretizedswap.cpp:184-197).
        """
        method = self._require_method()
        bond = DiscretizedDiscountBond()
        bond.initialize(method, self._fixed_pay_times[i])
        bond.rollback(self._time)
        fixed_coupon = self._arguments.fixed_coupons[i]
        coupon = fixed_coupon * bond.values
        if self._arguments.swap_type == SwapType.Payer:
            self._values = self._values - coupon
        else:
            self._values = self._values + coupon

    def _add_floating_coupon(self, i: int) -> None:
        """Discount the i-th floating-leg contribution back to ``self.time``.

        # C++ parity: ``DiscretizedSwap::addFloatingCoupon`` (discretizedswap.cpp:199-218).

        The contribution is the difference between the rolled-forward
        notional and the discounted-to-pay-time bond, plus the
        accrued spread weighted by the bond.
        """
        method = self._require_method()
        bond = DiscretizedDiscountBond()
        bond.initialize(method, self._floating_pay_times[i])
        bond.rollback(self._time)

        nominal = self._arguments.nominal
        qassert.require(
            nominal is not None,
            "non-constant nominals are not supported yet",
        )
        assert nominal is not None
        t = self._arguments.floating_accrual_times[i]
        spread = self._arguments.floating_spreads[i]
        accrued_spread = nominal * t * spread
        coupon = nominal * (1.0 - bond.values) + accrued_spread * bond.values
        if self._arguments.swap_type == SwapType.Payer:
            self._values = self._values + coupon
        else:
            self._values = self._values - coupon


__all__ = ["DiscretizedSwap"]
