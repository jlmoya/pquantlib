"""make_ois — free-function factory for OvernightIndexedSwap.

# C++ parity: ql/instruments/makeois.{hpp,cpp} ``MakeOIS``.

Free-function factory mirroring make_vanilla_swap's keyword-arg style.
Handles fair-rate fallback (build temp OIS @ 0% fixed, set engine,
read fair_rate) and default settlement-days inference for SONIA / CORRA
/ others.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.instruments.overnight_indexed_swap import OvernightIndexedSwap
from pquantlib.instruments.swap import SwapType
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule, allows_end_of_month
from pquantlib.time.time_unit import TimeUnit

_ZERO_PERIOD = Period(0, TimeUnit.Days)


def _default_settlement_days(idx_name: str) -> int:
    """Per-index spot-date convention.

    # C++ parity: ``MakeOIS::operator OvernightIndexedSwap()`` dispatches
    # by dynamic_pointer_cast<Sonia> -> 0, <Corra> -> 1, else 2
    # (makeois.cpp:60-69). PQuantLib uses index name as the proxy.
    """
    up = idx_name.upper()
    if "SONIA" in up:
        return 0
    if "CORRA" in up:
        return 1
    return 2


def make_ois(  # noqa: PLR0915 — mirrors C++ MakeOIS chain, deliberately one big resolution
    swap_tenor: Period,
    overnight_index: OvernightIndex,
    fixed_rate: float | None = None,
    forward_start: Period = _ZERO_PERIOD,
    *,
    nominal: float = 1.0,
    swap_type: SwapType = SwapType.Payer,
    settlement_days: int | None = None,
    effective_date: Date | None = None,
    termination_date: Date | None = None,
    payment_frequency: Frequency = Frequency.Annual,
    fixed_leg_payment_frequency: Frequency | None = None,
    overnight_leg_payment_frequency: Frequency | None = None,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following,
    payment_lag: int = 0,
    payment_calendar: Calendar | None = None,
    calendar: Calendar | None = None,
    fixed_leg_calendar: Calendar | None = None,
    overnight_leg_calendar: Calendar | None = None,
    fixed_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
    overnight_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
    fixed_termination_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
    overnight_termination_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
    fixed_rule: DateGeneration = DateGeneration.Backward,
    overnight_rule: DateGeneration = DateGeneration.Backward,
    end_of_month: bool | None = None,
    maturity_end_of_month: bool | None = None,
    fixed_leg_day_count: DayCounter | None = None,
    overnight_leg_spread: float = 0.0,
    discount_curve: YieldTermStructureProtocol | None = None,
    telescopic_value_dates: bool = False,
    evaluation_date: Date | None = None,
) -> OvernightIndexedSwap:
    """Construct a market-standard OvernightIndexedSwap.

    Mirrors C++ ``MakeOIS::operator OvernightIndexedSwap()`` (makeois.cpp:47-183).
    """
    fixed_cal = (
        fixed_leg_calendar
        if fixed_leg_calendar is not None
        else calendar
        if calendar is not None
        else overnight_index.fixing_calendar()
    )
    ov_cal = (
        overnight_leg_calendar
        if overnight_leg_calendar is not None
        else calendar
        if calendar is not None
        else overnight_index.fixing_calendar()
    )
    fixed_dc = (
        fixed_leg_day_count
        if fixed_leg_day_count is not None
        else overnight_index.day_counter()
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
            "make_ois: evaluation_date is required when effective_date is None",
        )
        assert evaluation_date is not None
        sd = (
            settlement_days
            if settlement_days is not None
            else _default_settlement_days(overnight_index.name())
        )
        ref = ov_cal.adjust(evaluation_date)
        spot = ov_cal.advance(ref, sd, TimeUnit.Days)
        start_date = spot + forward_start
        if forward_start.length < 0:
            start_date = ov_cal.adjust(start_date, BusinessDayConvention.Preceding)
        else:
            start_date = ov_cal.adjust(start_date, BusinessDayConvention.Following)

    # EOM resolution: caller may set explicit per-leg flags via the upcoming
    # extension; the simple flavour uses one ``end_of_month`` and a maturity
    # flag, falling back to "is start_date end-of-month?".
    if end_of_month is None:
        is_eom = ov_cal.is_end_of_month(start_date)
        fixed_eom = is_eom
        overnight_eom = is_eom
        maturity_eom = is_eom
    else:
        fixed_eom = end_of_month
        overnight_eom = end_of_month
        maturity_eom = maturity_end_of_month if maturity_end_of_month is not None else end_of_month

    # Resolve end_date.
    if termination_date is not None:
        end_date = termination_date
    else:
        end_date = start_date + swap_tenor
        if maturity_eom and allows_end_of_month(swap_tenor) and ov_cal.is_end_of_month(start_date):
            end_date = ov_cal.end_of_month(end_date)

    # Resolve per-leg payment frequencies.
    fixed_freq = (
        fixed_leg_payment_frequency
        if fixed_leg_payment_frequency is not None
        else payment_frequency
    )
    ov_freq = (
        overnight_leg_payment_frequency
        if overnight_leg_payment_frequency is not None
        else payment_frequency
    )

    fixed_rule_eff = DateGeneration.Zero if fixed_freq == Frequency.Once else fixed_rule
    ov_rule_eff = DateGeneration.Zero if ov_freq == Frequency.Once else overnight_rule

    fixed_sched = Schedule.from_rule(
        start_date, end_date, Period.from_frequency(fixed_freq), fixed_cal,
        fixed_convention, fixed_termination_convention,
        fixed_rule_eff, fixed_eom,
    )
    ov_sched = Schedule.from_rule(
        start_date, end_date, Period.from_frequency(ov_freq), ov_cal,
        overnight_convention, overnight_termination_convention,
        ov_rule_eff, overnight_eom,
    )

    used_fixed_rate = fixed_rate
    if used_fixed_rate is None:
        temp = OvernightIndexedSwap(
            swap_type, nominal, fixed_sched, 0.0, fixed_dc, overnight_index,
            spread=overnight_leg_spread, payment_lag=payment_lag,
            payment_adjustment=payment_adjustment, payment_calendar=payment_calendar,
            telescopic_value_dates=telescopic_value_dates,
            overnight_schedule=ov_sched,
        )
        ts = discount_curve if discount_curve is not None else _get_forecast_ts(overnight_index)
        qassert.require(
            ts is not None,
            f"make_ois: null term structure set to this instance of {overnight_index.name()}",
        )
        assert ts is not None
        temp.set_pricing_engine(
            DiscountingSwapEngine(ts, include_settlement_date_flows=False)
        )
        used_fixed_rate = temp.fair_rate()

    ois = OvernightIndexedSwap(
        swap_type, nominal, fixed_sched, used_fixed_rate, fixed_dc, overnight_index,
        spread=overnight_leg_spread, payment_lag=payment_lag,
        payment_adjustment=payment_adjustment, payment_calendar=payment_calendar,
        telescopic_value_dates=telescopic_value_dates,
        overnight_schedule=ov_sched,
    )
    ts = discount_curve if discount_curve is not None else _get_forecast_ts(overnight_index)
    if ts is not None:
        ois.set_pricing_engine(
            DiscountingSwapEngine(ts, include_settlement_date_flows=False)
        )
    return ois


def _get_forecast_ts(index: OvernightIndex) -> YieldTermStructureProtocol | None:
    """Duck-typed accessor for ``forecast_term_structure``."""
    get_ts = getattr(index, "forecast_term_structure", None)
    return get_ts() if get_ts is not None else None


__all__ = ["make_ois"]
