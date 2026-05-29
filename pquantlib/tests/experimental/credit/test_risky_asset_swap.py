"""Cross-validate RiskyAssetSwap + RiskyAssetSwapOption against C++.

Probe source: migration-harness/cpp/probes/cluster_w3d/probe.cpp
Reference:    migration-harness/references/cluster/w3d.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.credit.risky_asset_swap import RiskyAssetSwap
from pquantlib.experimental.credit.risky_asset_swap_option import (
    RiskyAssetSwapOption,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import MakeSchedule
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3d")


@pytest.fixture(autouse=True)
def _set_eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    """Pin the evaluation date to the curve reference date."""
    ObservableSettings().evaluation_date = Date.from_ymd(15, Month.January, 2024)


def _build_asw() -> RiskyAssetSwap:
    today = Date.from_ymd(15, Month.January, 2024)
    cal = TARGET()
    dc = Actual365Fixed()
    yts = FlatForward(today, SimpleQuote(0.03), dc,
                      Compounding.Continuous, Frequency.Annual)
    dts = FlatHazardRate(today, SimpleQuote(0.02), dc)
    start = cal.advance(today, 2, TimeUnit.Days)
    end = start + Period(5, TimeUnit.Years)

    fixed_sched = (
        MakeSchedule()
        .from_date(start)
        .to(end)
        .with_calendar(cal)
        .with_tenor(Period(1, TimeUnit.Years))
        .with_convention(BusinessDayConvention.Unadjusted)
        .backwards()
        .build()
    )
    float_sched = (
        MakeSchedule()
        .from_date(start)
        .to(end)
        .with_calendar(cal)
        .with_tenor(Period(6, TimeUnit.Months))
        .with_convention(BusinessDayConvention.Unadjusted)
        .backwards()
        .build()
    )
    return RiskyAssetSwap(
        fixed_payer=True,
        nominal=100.0,
        fixed_schedule=fixed_sched,
        float_schedule=float_sched,
        fixed_day_counter=dc,
        float_day_counter=dc,
        spread=0.0150,
        recovery_rate=0.4,
        yield_ts=yts,
        default_ts=dts,
        coupon=0.05,
    )


def test_risky_asset_swap_npv_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    """The Euler-integral recovery + closed-form bond price reproduce
    the C++ NPV. LOOSE tier: the daily Euler step accumulates ~1e-3
    rounding over the 5y schedule but stays well within 1e-8 due to
    the small magnitudes.
    """
    asw = _build_asw()
    ref = cpp_ref["risky_asset_swap"]
    tolerance.loose(asw.npv(), ref["npv"])
    tolerance.loose(asw.fair_spread(), ref["fair_spread"])
    tolerance.loose(asw.float_annuity(), ref["float_annuity"])
    tolerance.tight(asw.nominal(), ref["nominal"])
    tolerance.tight(asw.spread(), ref["spread"])


def test_risky_asset_swap_option_npv_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    """The Bachelier spread-option NPV matches C++ to LOOSE tier."""
    today = Date.from_ymd(15, Month.January, 2024)
    cal = TARGET()
    expiry = cal.advance(today, 1, TimeUnit.Years)
    asw = _build_asw()
    opt = RiskyAssetSwapOption(
        underlying=asw,
        expiry=expiry,
        market_spread=0.0200,
        spread_volatility=0.40,
    )
    ref = cpp_ref["risky_asw_option"]
    tolerance.loose(opt.npv(), ref["npv"])


def test_risky_asset_swap_fixed_payer_side_flip() -> None:
    """Switching ``fixed_payer`` flips the NPV sign exactly."""
    asw = _build_asw()
    npv_pay = asw.npv()

    today = Date.from_ymd(15, Month.January, 2024)
    cal = TARGET()
    dc = Actual365Fixed()
    yts = FlatForward(today, SimpleQuote(0.03), dc,
                      Compounding.Continuous, Frequency.Annual)
    dts = FlatHazardRate(today, SimpleQuote(0.02), dc)
    start = cal.advance(today, 2, TimeUnit.Days)
    end = start + Period(5, TimeUnit.Years)
    fixed_sched = (
        MakeSchedule()
        .from_date(start).to(end).with_calendar(cal)
        .with_tenor(Period(1, TimeUnit.Years))
        .with_convention(BusinessDayConvention.Unadjusted)
        .backwards().build()
    )
    float_sched = (
        MakeSchedule()
        .from_date(start).to(end).with_calendar(cal)
        .with_tenor(Period(6, TimeUnit.Months))
        .with_convention(BusinessDayConvention.Unadjusted)
        .backwards().build()
    )
    asw_recv = RiskyAssetSwap(
        fixed_payer=False,
        nominal=100.0,
        fixed_schedule=fixed_sched,
        float_schedule=float_sched,
        fixed_day_counter=dc,
        float_day_counter=dc,
        spread=0.0150,
        recovery_rate=0.4,
        yield_ts=yts,
        default_ts=dts,
        coupon=0.05,
    )
    tolerance.tight(asw_recv.npv(), -npv_pay)
