"""make_vanilla_swap — free-function factory for VanillaSwap.

# C++ parity: ql/instruments/makevanillaswap.{hpp,cpp} ``MakeVanillaSwap``.

The C++ class is a chained builder
(``MakeVanillaSwap(...).withNominal(N).withFixedLegTenor(T)...``) ending
with an ``operator VanillaSwap()`` conversion. PQuantLib L2-D
established the leg-builder pattern of folding the C++ chained-builder
into a single keyword-arg free function; this port follows the same
pattern.

The factory does the C++ MakeVanillaSwap's spot-date / end-date /
fixed-tenor / fixed-day-count inference (currency-driven defaults) plus
fair-rate fallback when ``fixed_rate`` is not supplied. The resulting
swap is wired with a ``DiscountingSwapEngine`` on either the explicitly-
passed discount curve or the IBOR index's forecast curve.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule, allows_end_of_month
from pquantlib.time.time_unit import TimeUnit

_ZERO_PERIOD = Period(0, TimeUnit.Days)


def _default_fixed_tenor(ccy_code: str, tenor: Period) -> Period:
    """Mirror the C++ MakeVanillaSwap currency-default switch (makevanillaswap.cpp:99-126)."""
    if ccy_code in ("EUR", "USD", "CHF", "SEK"):
        return Period(1, TimeUnit.Years)
    if ccy_code == "GBP":
        return Period(1, TimeUnit.Years) if tenor <= Period(1, TimeUnit.Years) else Period(6, TimeUnit.Months)
    if ccy_code in ("JPY",):
        return Period(6, TimeUnit.Months)
    if ccy_code in ("AUD",):
        return (
            Period(6, TimeUnit.Months)
            if tenor >= Period(4, TimeUnit.Years)
            else Period(3, TimeUnit.Months)
        )
    if ccy_code == "HKD":
        return Period(3, TimeUnit.Months)
    qassert.fail(f"unknown fixed leg default tenor for {ccy_code}")


def _default_fixed_day_count(ccy_code: str) -> DayCounter:
    """Mirror C++ MakeVanillaSwap day-counter defaults (makevanillaswap.cpp:142-157)."""
    if ccy_code == "USD":
        return Actual360()
    if ccy_code in ("EUR", "CHF", "SEK"):
        return Thirty360(Thirty360Convention.BondBasis)
    if ccy_code in ("GBP", "JPY", "AUD", "HKD", "THB"):
        return Actual365Fixed()
    qassert.fail(f"unknown fixed leg day counter for {ccy_code}")


def make_vanilla_swap(
    swap_tenor: Period,
    ibor_index: IborIndex,
    fixed_rate: float | None = None,
    forward_start: Period = _ZERO_PERIOD,
    *,
    nominal: float = 1.0,
    swap_type: SwapType = SwapType.Payer,
    settlement_days: int | None = None,
    effective_date: Date | None = None,
    termination_date: Date | None = None,
    fixed_leg_tenor: Period | None = None,
    fixed_leg_calendar: Calendar | None = None,
    fixed_leg_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
    fixed_leg_termination_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
    fixed_leg_rule: DateGeneration = DateGeneration.Backward,
    fixed_leg_end_of_month: bool = False,
    fixed_leg_day_count: DayCounter | None = None,
    floating_leg_tenor: Period | None = None,
    floating_leg_calendar: Calendar | None = None,
    floating_leg_convention: BusinessDayConvention | None = None,
    floating_leg_termination_convention: BusinessDayConvention | None = None,
    floating_leg_rule: DateGeneration = DateGeneration.Backward,
    floating_leg_end_of_month: bool = False,
    floating_leg_day_count: DayCounter | None = None,
    floating_leg_spread: float = 0.0,
    maturity_end_of_month: bool | None = None,
    payment_convention: BusinessDayConvention | None = None,
    discount_curve: YieldTermStructureProtocol | None = None,
    use_indexed_coupons: bool | None = None,
    evaluation_date: Date | None = None,
) -> VanillaSwap:
    """Construct a market-standard VanillaSwap.

    Mirrors C++ ``MakeVanillaSwap::operator VanillaSwap()`` (makevanillaswap.cpp:52-196).

    ``evaluation_date`` is REQUIRED if ``effective_date`` is not given,
    because PQuantLib doesn't expose a Settings global; the C++ code
    falls back to ``Settings.evaluationDate()``. Passing
    ``evaluation_date`` makes this explicit.
    """
    # Resolve calendars first (default to the IBOR index's fixing calendar).
    fixed_cal = (
        fixed_leg_calendar if fixed_leg_calendar is not None else ibor_index.fixing_calendar()
    )
    float_cal = (
        floating_leg_calendar
        if floating_leg_calendar is not None
        else ibor_index.fixing_calendar()
    )
    float_conv = (
        floating_leg_convention
        if floating_leg_convention is not None
        else ibor_index.business_day_convention()
    )
    float_term_conv = (
        floating_leg_termination_convention
        if floating_leg_termination_convention is not None
        else ibor_index.business_day_convention()
    )
    float_tenor = (
        floating_leg_tenor if floating_leg_tenor is not None else ibor_index.tenor()
    )
    float_dc = (
        floating_leg_day_count
        if floating_leg_day_count is not None
        else ibor_index.day_counter()
    )

    qassert.require(
        effective_date is None or settlement_days is None,
        "cannot set both an explicit effective date and settlement days; use one or the other",
    )

    # Resolve start_date.
    if effective_date is not None:
        start_date = effective_date
    else:
        qassert.require(
            evaluation_date is not None,
            "make_vanilla_swap: evaluation_date is required when effective_date is None",
        )
        assert evaluation_date is not None
        ref = float_cal.adjust(evaluation_date)
        sd = settlement_days if settlement_days is not None else ibor_index.fixing_days()
        spot = float_cal.advance(ref, sd, TimeUnit.Days)
        start_date = spot + forward_start
        if forward_start.length < 0:
            start_date = float_cal.adjust(start_date, BusinessDayConvention.Preceding)
        elif forward_start.length > 0:
            start_date = float_cal.adjust(start_date, BusinessDayConvention.Following)

    # Resolve end_date.
    if termination_date is not None:
        end_date = termination_date
    else:
        end_date = start_date + swap_tenor
        m_eom = maturity_end_of_month if maturity_end_of_month is not None else floating_leg_end_of_month
        if m_eom and allows_end_of_month(swap_tenor) and float_cal.is_end_of_month(start_date):
            end_date = float_cal.end_of_month(end_date)

    # Currency-driven fixed-leg defaults.
    ccy = ibor_index.currency()
    ccy_code: str = getattr(ccy, "code", "EUR")
    eff_fixed_tenor = (
        fixed_leg_tenor if fixed_leg_tenor is not None else _default_fixed_tenor(ccy_code, swap_tenor)
    )
    eff_fixed_dc = (
        fixed_leg_day_count
        if fixed_leg_day_count is not None
        else _default_fixed_day_count(ccy_code)
    )

    # Resolve payment_convention default — C++ defers to the float schedule's BDC.
    pay_conv = payment_convention if payment_convention is not None else float_conv

    fixed_sched = Schedule.from_rule(
        start_date, end_date, eff_fixed_tenor, fixed_cal,
        fixed_leg_convention, fixed_leg_termination_convention,
        fixed_leg_rule, fixed_leg_end_of_month,
    )
    float_sched = Schedule.from_rule(
        start_date, end_date, float_tenor, float_cal,
        float_conv, float_term_conv,
        floating_leg_rule, floating_leg_end_of_month,
    )

    used_fixed_rate = fixed_rate
    if used_fixed_rate is None:
        # Fair-rate fallback: build a temporary swap with rate 0.0 + the index's
        # forecast curve, set its engine, read fair_rate. C++ creates a 100-nominal
        # temp; we mirror.
        temp = VanillaSwap(
            swap_type, 100.0, fixed_sched, 0.0, eff_fixed_dc,
            float_sched, ibor_index, floating_leg_spread, float_dc,
            payment_convention=pay_conv, use_indexed_coupons=use_indexed_coupons,
        )
        ts = discount_curve if discount_curve is not None else ibor_index.forecast_term_structure()
        qassert.require(
            ts is not None,
            f"make_vanilla_swap: null term structure set to this instance of {ibor_index.name()}",
        )
        assert ts is not None
        temp.set_pricing_engine(
            DiscountingSwapEngine(ts, include_settlement_date_flows=False)
        )
        used_fixed_rate = temp.fair_rate()

    swap = VanillaSwap(
        swap_type, nominal, fixed_sched, used_fixed_rate, eff_fixed_dc,
        float_sched, ibor_index, floating_leg_spread, float_dc,
        payment_convention=pay_conv, use_indexed_coupons=use_indexed_coupons,
    )

    ts = discount_curve if discount_curve is not None else ibor_index.forecast_term_structure()
    if ts is not None:
        swap.set_pricing_engine(
            DiscountingSwapEngine(ts, include_settlement_date_flows=False)
        )

    return swap


__all__ = ["make_vanilla_swap"]
