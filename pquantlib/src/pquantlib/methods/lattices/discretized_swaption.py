"""DiscretizedSwaption — option on a discretized swap.

# C++ parity: ql/pricingengines/swaption/discretizedswaption.{hpp,cpp} (v1.42.1).

Owns a ``DiscretizedSwap`` underlying plus the swaption's exercise
schedule. The date-snapping logic from the C++ source is preserved:
when an exercise date falls within one week of a coupon date, the
coupon date is snapped to the exercise date so that the lattice's
``is_on_time`` check fires for both.

The discretized swaption inherits the standard
:class:`DiscretizedOption` ``post_adjust_values_impl`` which rolls
the underlying back and applies ``max(underlying, current)``.

Snapping algorithm (C++ ``prepareSwaptionWithSnappedDates``,
discretizedswaption.cpp:77-125):

  * For each exercise date and each coupon date (fixed + floating),
    if the unadjusted coupon date is within one week of the exercise
    date, snap the coupon date to the exercise date.
  * If the original coupon was *before* the exercise (i.e. snapping
    moves it forward), flip its adjustment to ``post``.
  * Build a fresh swap with the snapped schedules so the
    ``DiscretizedSwap`` ctor sees a consistent set of times.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.instruments.swaption import SwaptionArguments
from pquantlib.methods.lattices.discretized_asset import CouponAdjustment
from pquantlib.methods.lattices.discretized_option import DiscretizedOption
from pquantlib.methods.lattices.discretized_swap import DiscretizedSwap
from pquantlib.time.date import Date as _Date

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.instruments.fixed_vs_floating_swap import (
        FixedVsFloatingSwapArguments,
    )
    from pquantlib.time.date import Date


class DiscretizedSwaption(DiscretizedOption):
    """Discretized European/Bermudan swaption on a vanilla swap.

    # C++ parity: ``class DiscretizedSwaption`` (discretizedswaption.hpp:34-52).
    """

    def __init__(
        self,
        args: SwaptionArguments,
        reference_date: Date,
        day_counter: DayCounter,
    ) -> None:
        # # C++ parity: discretizedswaption.cpp:39-70.
        # 1) Snap the swap's coupon dates to nearby exercise dates.
        # 2) Build the snapped Swaption arguments + adjustments.
        # 3) Compute exercise times.
        # 4) Compute the maximum payment time (last underlying time).
        # 5) Construct the DiscretizedSwap underlying.

        assert args.exercise is not None
        assert args.swap is not None

        snapped_args, fixed_adj, floating_adj = self._prepare_swaption_with_snapped_dates(
            args
        )

        # Exercise times (sorted by underlying exercise schedule).
        exercise_times = [
            day_counter.year_fraction(reference_date, d)
            for d in args.exercise.dates()
        ]

        # Last payment time bounds the rollback. The pay-date lists are
        # typed ``list[object]`` (carrier polymorphism on SwapArguments);
        # narrow each entry to Date for the year-fraction call.
        last_fixed_pay = snapped_args.fixed_pay_dates[-1]
        last_floating_pay = snapped_args.floating_pay_dates[-1]
        assert isinstance(last_fixed_pay, _Date)
        assert isinstance(last_floating_pay, _Date)
        last_fixed_payment = day_counter.year_fraction(reference_date, last_fixed_pay)
        last_floating_payment = day_counter.year_fraction(
            reference_date, last_floating_pay
        )
        self._last_payment: float = max(last_fixed_payment, last_floating_payment)

        # Build the underlying swap discretization with the snapped args.
        underlying = DiscretizedSwap(
            snapped_args,
            reference_date,
            day_counter,
            fixed_coupon_adjustments=fixed_adj,
            floating_coupon_adjustments=floating_adj,
        )

        super().__init__(
            underlying=underlying,
            exercise_type=args.exercise.type(),
            exercise_times=exercise_times,
        )

    # --- DiscretizedAsset interface --------------------------------------

    def reset(self, size: int) -> None:
        """Initialise the underlying at ``last_payment`` then zero-init.

        # C++ parity: ``DiscretizedSwaption::reset`` (discretizedswaption.cpp:72-75).
        """
        method = self._require_method()
        self._underlying.initialize(method, self._last_payment)
        super().reset(size)

    # --- snapping algorithm ----------------------------------------------

    @staticmethod
    def _within_previous_week(d1: Date, d2: Date) -> bool:
        # # C++ parity: ``withinPreviousWeek`` (anonymous-ns, discretizedswaption.cpp:30).
        return d1 - 7 <= d2 <= d1

    @staticmethod
    def _within_next_week(d1: Date, d2: Date) -> bool:
        # # C++ parity: ``withinNextWeek`` (discretizedswaption.cpp:32).
        return d1 <= d2 <= d1 + 7

    @classmethod
    def _within_one_week(cls, d1: Date, d2: Date) -> bool:
        # # C++ parity: ``withinOneWeek`` (discretizedswaption.cpp:34-36).
        return cls._within_previous_week(d1, d2) or cls._within_next_week(d1, d2)

    @classmethod
    def _prepare_swaption_with_snapped_dates(
        cls,
        args: SwaptionArguments,
    ) -> tuple[FixedVsFloatingSwapArguments, list[CouponAdjustment], list[CouponAdjustment]]:
        """Snap coupon dates to exercise dates if within one week.

        # C++ parity: ``DiscretizedSwaption::prepareSwaptionWithSnappedDates``
        # (discretizedswaption.cpp:77-125).

        Returns the snapped ``SwaptionArguments`` plus the per-coupon
        adjustment vectors for the fixed and floating legs.
        """
        assert args.swap is not None
        assert args.exercise is not None
        swap = args.swap

        fixed_schedule = swap.fixed_schedule()
        floating_schedule = swap.floating_schedule()
        fixed_dates = list(fixed_schedule.dates)
        float_dates = list(floating_schedule.dates)

        n_fixed = len(swap.fixed_leg())
        n_float = len(swap.floating_leg())
        fixed_adj: list[CouponAdjustment] = [CouponAdjustment.Pre] * n_fixed
        floating_adj: list[CouponAdjustment] = [CouponAdjustment.Pre] * n_float

        for exercise_d in args.exercise.dates():
            # Walk all but the last fixed coupon date (C++ uses
            # ``fixed_dates.size() - 1`` — the terminal pay date is
            # excluded from snapping).
            for j in range(len(fixed_dates) - 1):
                d_un = fixed_dates[j]
                assert isinstance(d_un, _Date)
                if exercise_d != d_un and cls._within_one_week(exercise_d, d_un):
                    fixed_dates[j] = exercise_d
                    if cls._within_previous_week(exercise_d, d_un):
                        fixed_adj[j] = CouponAdjustment.Post

            for j in range(len(float_dates) - 1):
                d_un = float_dates[j]
                assert isinstance(d_un, _Date)
                if exercise_d != d_un and cls._within_one_week(exercise_d, d_un):
                    float_dates[j] = exercise_d
                    if cls._within_previous_week(exercise_d, d_un):
                        floating_adj[j] = CouponAdjustment.Post

        # # C++ parity: discretizedswaption.cpp:113-124 — re-build the
        # # snapped Swaption.
        # PQuantLib divergence: the C++ source rebuilds a VanillaSwap +
        # SnappedSwaption and calls ``setupArguments``. We take the
        # simpler path: copy the existing ``args`` and rewrite the
        # fixed/floating pay-date vectors. The reset-date vectors stay
        # in lock-step with pay because the C++ Schedule(dates) ctor
        # treats consecutive entries as one period.
        snapped = SwaptionArguments()
        snapped.swap_type = args.swap_type
        snapped.nominal = args.nominal
        snapped.fixed_pay_dates = [
            fixed_dates[j + 1] for j in range(n_fixed)
        ] if len(fixed_dates) > n_fixed else list(args.fixed_pay_dates)
        snapped.fixed_reset_dates = [
            fixed_dates[j] for j in range(n_fixed)
        ] if len(fixed_dates) > n_fixed else list(args.fixed_reset_dates)
        snapped.fixed_coupons = list(args.fixed_coupons)
        snapped.fixed_nominals = list(args.fixed_nominals)
        snapped.floating_pay_dates = [
            float_dates[j + 1] for j in range(n_float)
        ] if len(float_dates) > n_float else list(args.floating_pay_dates)
        snapped.floating_reset_dates = [
            float_dates[j] for j in range(n_float)
        ] if len(float_dates) > n_float else list(args.floating_reset_dates)
        snapped.floating_fixing_dates = list(args.floating_fixing_dates)
        snapped.floating_accrual_times = list(args.floating_accrual_times)
        snapped.floating_coupons = list(args.floating_coupons)
        snapped.floating_nominals = list(args.floating_nominals)
        snapped.floating_spreads = list(args.floating_spreads)
        snapped.swap = args.swap
        snapped.settlement_type = args.settlement_type
        snapped.settlement_method = args.settlement_method
        snapped.exercise = args.exercise
        snapped.payoff = args.payoff
        return snapped, fixed_adj, floating_adj


__all__ = ["DiscretizedSwaption"]
