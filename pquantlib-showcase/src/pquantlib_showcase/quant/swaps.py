"""Interest-rate swaps: vanilla fixed-vs-float and overnight-indexed (OIS)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.ibor.sofr import Sofr
from pquantlib.instruments.overnight_indexed_swap import OvernightIndexedSwap
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

from .common import flat_curve, pin_evaluation_date, reference_date

_FIXED_DC = Thirty360(Convention.BondBasis)


@dataclass(frozen=True, slots=True)
class SwapResult:
    npv: float
    fair_rate: float
    fixed_leg_npv: float
    floating_leg_npv: float
    fixed_leg_bps: float


def _build_vanilla(notional: float, years: int, fixed_rate: float, curve_rate: float) -> VanillaSwap:
    ref = reference_date()
    pin_evaluation_date(ref)
    curve = flat_curve(curve_rate, ref)
    idx = Euribor(Period(6, TimeUnit.Months), curve)
    cal = TARGET()
    start = cal.advance(ref, 2, TimeUnit.Days)
    end = start + Period(years, TimeUnit.Years)
    fixed = Schedule.from_rule(
        start,
        end,
        Period(1, TimeUnit.Years),
        cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward,
        False,
    )
    flt = Schedule.from_rule(
        start,
        end,
        Period(6, TimeUnit.Months),
        cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward,
        False,
    )
    swap = VanillaSwap(
        SwapType.Payer, notional, fixed, fixed_rate, _FIXED_DC, flt, idx, 0.0, idx.day_counter()
    )
    swap.set_pricing_engine(DiscountingSwapEngine(curve))
    return swap


def price_vanilla_swap(notional: float, years: int, fixed_rate: float, curve_rate: float) -> SwapResult:
    """Price a payer vanilla swap (pay fixed, receive Euribor 6M)."""
    swap = _build_vanilla(notional, years, fixed_rate, curve_rate)
    return SwapResult(
        npv=swap.npv(),
        fair_rate=swap.fair_rate(),
        fixed_leg_npv=swap.fixed_leg_npv(),
        floating_leg_npv=swap.floating_leg_npv(),
        fixed_leg_bps=swap.fixed_leg_bps(),
    )


def price_ois(notional: float, years: int, fixed_rate: float, curve_rate: float) -> SwapResult:
    """Price an overnight-indexed swap (vs SOFR)."""
    ref = reference_date()
    pin_evaluation_date(ref)
    curve = flat_curve(curve_rate, ref)
    cal = TARGET()
    start = cal.advance(ref, 2, TimeUnit.Days)
    end = start + Period(years, TimeUnit.Years)
    sched = Schedule.from_rule(
        start,
        end,
        Period(1, TimeUnit.Years),
        cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward,
        False,
    )
    from pquantlib.daycounters.actual_360 import Actual360

    ois = OvernightIndexedSwap(SwapType.Payer, notional, sched, fixed_rate, Actual360(), Sofr(curve))
    ois.set_pricing_engine(DiscountingSwapEngine(curve))
    return SwapResult(
        npv=ois.npv(),
        fair_rate=ois.fair_rate(),
        fixed_leg_npv=ois.fixed_leg_npv(),
        floating_leg_npv=ois.overnight_leg_npv(),
        fixed_leg_bps=ois.fixed_leg_bps(),
    )


def npv_vs_fixed_rate(
    notional: float, years: int, curve_rate: float, rate_grid: list[float]
) -> tuple[list[float], list[float]]:
    """Swap NPV as the contractual fixed rate sweeps ``rate_grid``."""
    npvs = [price_vanilla_swap(notional, years, r, curve_rate).npv for r in rate_grid]
    return list(rate_grid), npvs


def default_fixed_rate_grid(low: float = 0.0, high: float = 0.08, n: int = 50) -> list[float]:
    return list(np.linspace(low, high, n))
