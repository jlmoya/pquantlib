"""AssetSwap — bullet bond versus floating (ibor or overnight) swap.

# C++ parity: ql/instruments/assetswap.{hpp,cpp} (v1.43).

The instrument pairs a bond's remaining cashflows against a floating leg.
In the *par* asset swap (``par_asset_swap=True``) the floating leg carries an
upfront payment of ``(dirty_price - 100)/100 * notional`` on the swap start
date plus a notional back-payment at the final date; in the *market* asset
swap the floating notional is scaled by ``dirty_price/100`` and only the final
notional exchange is added.  For the mechanics see "Introduction to Asset
Swap", Lehman Brothers European Fixed Income Research, January 2000.

``bond_clean_price`` must be the (forward) clean price at the floating
schedule's start date.

Python port notes:

- C++ null sentinels become ``None``: an empty ``Schedule()`` →
  ``float_schedule=None``, an empty ``DayCounter()`` →
  ``floating_day_count=None``, ``Null<Real>()`` → ``non_par_repayment=None``,
  and ``Date()`` → ``deal_maturity=None``.
- The legs are built through the ``ibor_leg`` / ``overnight_leg`` free
  functions, which are this port's spelling of the C++ ``IborLeg`` /
  ``OvernightLeg`` fluent builders.
- ``AssetSwap`` prices through ``DiscountingSwapEngine`` (which produces a
  plain ``SwapResults``); ``AssetSwapArguments`` / ``AssetSwapResults`` exist
  for engines that want the disaggregated coupon vectors, exactly as in C++.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.coupon import Coupon
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.cashflows.overnight_leg import overnight_leg
from pquantlib.cashflows.simple_cash_flow import SimpleCashFlow
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.instruments.bond import Bond
from pquantlib.instruments.swap import Swap, SwapArguments, SwapResults
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.schedule import Schedule

_BASIS_POINT = 1.0e-4


class AssetSwapArguments(SwapArguments):
    """Disaggregated coupon vectors for asset-swap-aware engines.

    # C++ parity: ``AssetSwap::arguments`` (assetswap.hpp:104-118).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fixed_reset_dates: list[Date] = []
        self.fixed_pay_dates: list[Date] = []
        self.fixed_coupons: list[float] = []
        self.floating_accrual_times: list[float] = []
        self.floating_reset_dates: list[Date] = []
        self.floating_fixing_dates: list[Date] = []
        self.floating_pay_dates: list[Date] = []
        self.floating_spreads: list[float] = []

    def validate(self) -> None:
        """# C++ parity: ``AssetSwap::arguments::validate`` (assetswap.cpp:284-306)."""
        super().validate()
        qassert.require(
            len(self.fixed_reset_dates) == len(self.fixed_pay_dates),
            "number of fixed start dates different from number of fixed payment dates",
        )
        qassert.require(
            len(self.fixed_pay_dates) == len(self.fixed_coupons),
            "number of fixed payment dates different from number of fixed coupon amounts",
        )
        qassert.require(
            len(self.floating_reset_dates) == len(self.floating_pay_dates),
            "number of floating start dates different from number of floating payment dates",
        )
        qassert.require(
            len(self.floating_fixing_dates) == len(self.floating_pay_dates),
            "number of floating fixing dates different from number of floating payment dates",
        )
        qassert.require(
            len(self.floating_accrual_times) == len(self.floating_pay_dates),
            "number of floating accrual times different from number of floating payment dates",
        )
        qassert.require(
            len(self.floating_spreads) == len(self.floating_pay_dates),
            "number of floating spreads different from number of floating payment dates",
        )


class AssetSwapResults(SwapResults):
    """# C++ parity: ``AssetSwap::results`` (assetswap.hpp:120-126)."""

    def __init__(self) -> None:
        super().__init__()
        self.fair_spread: float | None = None
        self.fair_clean_price: float | None = None
        self.fair_non_par_repayment: float | None = None

    def reset(self) -> None:
        super().reset()
        self.fair_spread = None
        self.fair_clean_price = None
        self.fair_non_par_repayment = None


class AssetSwap(Swap):
    """Bullet bond versus floating swap.

    # C++ parity: ``AssetSwap`` (assetswap.hpp:41-99, assetswap.cpp:36-186).
    """

    def __init__(
        self,
        pay_bond_coupon: bool,
        bond: Bond,
        bond_clean_price: float,
        ibor_index: IborIndex,
        spread: float,
        float_schedule: Schedule | None = None,
        floating_day_count: DayCounter | None = None,
        par_asset_swap: bool = True,
        gearing: float = 1.0,
        non_par_repayment: float | None = None,
        deal_maturity: Date | None = None,
    ) -> None:
        # # C++ parity: ``AssetSwap::AssetSwap`` (assetswap.cpp:36-186).
        super().__init__(n_legs=2)
        self._bond: Bond = bond
        self._bond_clean_price: float = bond_clean_price
        self._non_par_repayment: float = 0.0  # finalised below
        self._spread: float = spread
        self._par_swap: bool = par_asset_swap
        self._fair_spread: float | None = None
        self._fair_clean_price: float | None = None
        self._fair_non_par_repayment: float | None = None

        overnight = ibor_index if isinstance(ibor_index, OvernightIndex) else None
        if overnight is not None:
            qassert.require(
                float_schedule is not None and not float_schedule.empty(),
                "floating schedule is needed when using an overnight index",
            )

        if float_schedule is None or float_schedule.empty():
            schedule = Schedule.from_rule(
                bond.settlement_date(),
                bond.maturity_date(),
                ibor_index.tenor(),
                ibor_index.fixing_calendar(),
                ibor_index.business_day_convention(),
                ibor_index.business_day_convention(),
                DateGeneration.Backward,
                False,  # end_of_month
            )
        else:
            schedule = float_schedule

        if deal_maturity is None or deal_maturity == Date():
            deal_maturity = schedule.back()
        qassert.require(
            deal_maturity <= schedule.back(),
            f"deal maturity {deal_maturity} cannot be later than (adjusted) bond maturity {schedule.back()}",
        )
        qassert.require(
            deal_maturity > schedule.front(),
            f"deal maturity {deal_maturity} must be later than swap start date {schedule.front()}",
        )

        # The following might become an input parameter (as in C++).
        payment_adjustment = BusinessDayConvention.Following

        final_date = schedule.calendar.adjust(deal_maturity, payment_adjustment)
        schedule = schedule.until(final_date)

        # bond_clean_price must be the (forward) clean price at the floating
        # schedule start date.
        self._upfront_date: Date = schedule.start_date
        dirty_price = bond_clean_price + bond.accrued_amount(self._upfront_date)

        notional = bond.notional(self._upfront_date)
        # In the market asset swap the bond is purchased in return for payment
        # of the full price, so the floating notional is scaled by it.
        if not self._par_swap:
            notional *= dirty_price / 100.0

        self._build_bond_leg(bond, deal_maturity, final_date, non_par_repayment)
        self._build_floating_leg(
            schedule=schedule,
            ibor_index=ibor_index,
            overnight=overnight,
            notional=notional,
            dirty_price=dirty_price,
            final_date=final_date,
            floating_day_count=floating_day_count,
            payment_adjustment=payment_adjustment,
            gearing=gearing,
            spread=spread,
        )

        # ---- registration and sides -----------------------------------
        for leg in self._legs:
            for cf in leg:
                cf.register_with(self)

        if pay_bond_coupon:
            self._payer = [-1.0, +1.0]
        else:
            self._payer = [+1.0, -1.0]

    # --- leg construction ----------------------------------------------

    def _build_bond_leg(
        self,
        bond: Bond,
        deal_maturity: Date,
        final_date: Date,
        non_par_repayment: float | None,
    ) -> None:
        """# C++ parity: the ``/******** Bond leg ********/`` block
        (assetswap.cpp:96-131)."""
        bond_leg = bond.cashflows()
        qassert.require(len(bond_leg) > 0, "no cashflows from bond")

        # A cashflow ON the upfront date must be discarded.
        include_on_upfront_date = False

        # Add coupons for the time being, not the redemption.
        i = 0
        while i < len(bond_leg) - 1 and bond_leg[i].date() <= deal_maturity:
            if not bond_leg[i].has_occurred(self._upfront_date, include_on_upfront_date):
                self._legs[0].append(bond_leg[i])
            i += 1

        # If we're skipping a cashflow before the redemption and it is a
        # coupon, add the accrued coupon instead.
        if i < len(bond_leg) - 1:
            skipped = bond_leg[i]
            if isinstance(skipped, Coupon):
                accrued_amount = skipped.accrued_amount(deal_maturity)
                self._legs[0].append(SimpleCashFlow(accrued_amount, final_date))

        # Add the redemption, or whatever the final payment is.
        if non_par_repayment is None:
            redemption = bond_leg[-1]
            self._legs[0].append(SimpleCashFlow(redemption.amount(), final_date))
            self._non_par_repayment = 100.0
        else:
            self._legs[0].append(SimpleCashFlow(non_par_repayment, final_date))
            self._non_par_repayment = non_par_repayment

    def _build_floating_leg(
        self,
        *,
        schedule: Schedule,
        ibor_index: IborIndex,
        overnight: OvernightIndex | None,
        notional: float,
        dirty_price: float,
        final_date: Date,
        floating_day_count: DayCounter | None,
        payment_adjustment: BusinessDayConvention,
        gearing: float,
        spread: float,
    ) -> None:
        """# C++ parity: the ``/******** Floating leg ********/`` block
        (assetswap.cpp:133-172)."""
        if overnight is not None:
            self._legs[1] = list(
                overnight_leg(
                    schedule,
                    overnight,
                    [notional],
                    payment_day_counter=floating_day_count,
                    payment_adjustment=payment_adjustment,
                    gearings=gearing,
                    spreads=spread,
                )
            )
        else:
            self._legs[1] = list(
                ibor_leg(
                    schedule,
                    ibor_index,
                    [notional],
                    payment_day_counter=floating_day_count,
                    payment_adjustment=payment_adjustment,
                    gearings=gearing,
                    spreads=spread,
                )
            )

        if self._par_swap:
            upfront = (dirty_price - 100.0) / 100.0 * notional
            self._legs[1].insert(0, SimpleCashFlow(upfront, self._upfront_date))
            # Back-payment (accounts for a non-par redemption, if any).
            self._legs[1].append(SimpleCashFlow(notional, final_date))
        else:
            # Final notional exchange.
            self._legs[1].append(SimpleCashFlow(notional, final_date))

    # --- inspectors ----------------------------------------------------

    def par_swap(self) -> bool:
        return self._par_swap

    def spread(self) -> float:
        return self._spread

    def clean_price(self) -> float:
        return self._bond_clean_price

    def non_par_repayment(self) -> float:
        return self._non_par_repayment

    def bond(self) -> Bond:
        return self._bond

    def pay_bond_coupon(self) -> bool:
        return self._payer[0] == -1.0

    def bond_leg(self) -> list[CashFlow]:
        return self._legs[0]

    def floating_leg(self) -> list[CashFlow]:
        return self._legs[1]

    # --- results -------------------------------------------------------

    def fair_spread(self) -> float:
        """# C++ parity: ``AssetSwap::fairSpread`` (assetswap.cpp:222-233)."""
        self.calculate()
        if self._fair_spread is not None:
            return self._fair_spread
        if len(self._leg_bps) > 1 and self._leg_bps[1] is not None:
            assert self._npv is not None
            self._fair_spread = self._spread - self._npv / self._leg_bps[1] * _BASIS_POINT
            return self._fair_spread
        qassert.fail("fair spread not available")

    def floating_leg_bps(self) -> float:
        """# C++ parity: ``AssetSwap::floatingLegBPS`` (assetswap.cpp:235-240)."""
        self.calculate()
        qassert.require(
            len(self._leg_bps) > 1 and self._leg_bps[1] is not None,
            "floating-leg BPS not available",
        )
        bps = self._leg_bps[1]
        assert bps is not None
        return bps

    def floating_leg_npv(self) -> float:
        """# C++ parity: ``AssetSwap::floatingLegNPV`` (assetswap.cpp:242-247)."""
        self.calculate()
        qassert.require(
            len(self._leg_npv) > 1 and self._leg_npv[1] is not None,
            "floating-leg NPV not available",
        )
        npv = self._leg_npv[1]
        assert npv is not None
        return npv

    def fair_clean_price(self) -> float:
        """# C++ parity: ``AssetSwap::fairCleanPrice`` (assetswap.cpp:249-269)."""
        self.calculate()
        if self._fair_clean_price is not None:
            return self._fair_clean_price
        qassert.require(
            self._start_discounts[1] is not None,
            "fair clean price not available for seasoned deal",
        )
        notional = self._bond.notional(self._upfront_date)
        if self._par_swap:
            start_discount = self._start_discounts[1]
            assert start_discount is not None
            assert self._npv is not None
            assert self._npv_date_discount is not None
            float_payer = self._payer[1]
            self._fair_clean_price = self._bond_clean_price - (
                float_payer * self._npv * self._npv_date_discount / start_discount
            ) / (notional / 100.0)
        else:
            accrued_amount = self._bond.accrued_amount(self._upfront_date)
            dirty_price = self._bond_clean_price + accrued_amount
            bond_leg_npv = self._leg_npv[0]
            float_leg_npv = self._leg_npv[1]
            assert bond_leg_npv is not None
            assert float_leg_npv is not None
            fair_dirty_price = -bond_leg_npv / float_leg_npv * dirty_price
            self._fair_clean_price = fair_dirty_price - accrued_amount
        return self._fair_clean_price

    def fair_non_par_repayment(self) -> float:
        """# C++ parity: ``AssetSwap::fairNonParRepayment`` (assetswap.cpp:271-284)."""
        self.calculate()
        if self._fair_non_par_repayment is not None:
            return self._fair_non_par_repayment
        qassert.require(
            self._end_discounts[1] is not None,
            "fair non par repayment not available for expired leg",
        )
        end_discount = self._end_discounts[1]
        assert end_discount is not None
        assert self._npv is not None
        assert self._npv_date_discount is not None
        notional = self._bond.notional(self._upfront_date)
        bond_payer = self._payer[0]
        self._fair_non_par_repayment = self._non_par_repayment - (
            bond_payer * self._npv * self._npv_date_discount / end_discount
        ) / (notional / 100.0)
        return self._fair_non_par_repayment

    # --- engine plumbing -----------------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Fill an asset-swap-aware argument carrier; no-op for a plain swap engine.

        # C++ parity: ``AssetSwap::setupArguments`` (assetswap.cpp:188-220).

        Note that no engine shipped with C++ v1.43 declares
        ``AssetSwap::arguments``, and the C++ body cannot work if one did: it
        ``dynamic_pointer_cast``s every bond-leg flow to ``FixedRateCoupon``
        and every floating-leg flow to ``FloatingRateCoupon`` and dereferences
        the result without a null check, while both legs always end in a
        ``SimpleCashFlow`` (redemption / back-payment, and the upfront on a par
        swap).  Rather than reproduce that undefined behaviour, the port raises
        a ``LibraryException`` naming the flow — the carriers exist for API
        parity and for ``fetch_results``, which *is* reachable.
        """
        super().setup_arguments(args)
        if not isinstance(args, AssetSwapArguments):
            # It's a plain swap engine.
            return

        fixed_coupons: Sequence[CashFlow] = self.bond_leg()
        args.fixed_reset_dates = []
        args.fixed_pay_dates = []
        args.fixed_coupons = []
        for cf in fixed_coupons:
            qassert.require(
                isinstance(cf, Coupon),
                "bond-leg flow is not a coupon; AssetSwap::arguments cannot "
                "represent it (C++ dereferences a null cast here)",
            )
            assert isinstance(cf, Coupon)
            args.fixed_pay_dates.append(cf.date())
            args.fixed_reset_dates.append(cf.accrual_start_date())
            args.fixed_coupons.append(cf.amount())

        floating_coupons: Sequence[CashFlow] = self.floating_leg()
        args.floating_reset_dates = []
        args.floating_pay_dates = []
        args.floating_fixing_dates = []
        args.floating_accrual_times = []
        args.floating_spreads = []
        for cf in floating_coupons:
            qassert.require(
                isinstance(cf, FloatingRateCoupon),
                "floating-leg flow is not a floating-rate coupon; "
                "AssetSwap::arguments cannot represent it "
                "(C++ dereferences a null cast here)",
            )
            assert isinstance(cf, FloatingRateCoupon)
            args.floating_reset_dates.append(cf.accrual_start_date())
            args.floating_pay_dates.append(cf.date())
            args.floating_fixing_dates.append(cf.fixing_date())
            args.floating_accrual_times.append(cf.accrual_period())
            args.floating_spreads.append(cf.spread())

    def fetch_results(self, results: PricingEngineResults) -> None:
        """# C++ parity: ``AssetSwap::fetchResults`` (assetswap.cpp:293-305)."""
        super().fetch_results(results)
        if isinstance(results, AssetSwapResults):
            self._fair_spread = results.fair_spread
            self._fair_clean_price = results.fair_clean_price
            self._fair_non_par_repayment = results.fair_non_par_repayment
        else:
            self._fair_spread = None
            self._fair_clean_price = None
            self._fair_non_par_repayment = None

    def setup_expired(self) -> None:
        """# C++ parity: ``AssetSwap::setupExpired`` (assetswap.cpp:286-291)."""
        super().setup_expired()
        self._fair_spread = None
        self._fair_clean_price = None
        self._fair_non_par_repayment = None


__all__ = ["AssetSwap", "AssetSwapArguments", "AssetSwapResults"]
