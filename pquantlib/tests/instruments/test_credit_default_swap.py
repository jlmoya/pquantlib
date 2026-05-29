"""Cross-validate CreditDefaultSwap + MidPoint / Integral CDS engines against C++.

Probe source: migration-harness/cpp/probes/cluster_l8b/probe.cpp
Reference:    migration-harness/references/cluster/l8b.json (key: "cds_engine")
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.instruments.credit_default_swap import CreditDefaultSwap, ProtectionSide
from pquantlib.pricingengines.credit.integral_cds_engine import IntegralCdsEngine
from pquantlib.pricingengines.credit.midpoint_cds_engine import MidPointCdsEngine
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.weekends_only import WeekendsOnly
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l8b")["cds_engine"]


@pytest.fixture(scope="module")
def ref_date() -> Date:
    return Date.from_ymd(15, Month.June, 2026)


@pytest.fixture
def cds_setup(
    ref_date: Date,
) -> tuple[
    CreditDefaultSwap,
    FlatHazardRate,
    FlatForward,
]:
    """Build the canonical 5y/200bps/lambda=0.02/3% CDS used in the probe."""
    cal = WeekendsOnly()
    bdc = BusinessDayConvention.Following
    dc365 = Actual365Fixed()
    dc360 = Actual360()
    maturity = ref_date + Period(5, TimeUnit.Years)
    schedule = Schedule.from_rule(
        ref_date, maturity, Period.from_frequency(Frequency.Quarterly),
        cal, bdc, bdc, DateGeneration.TwentiethIMM, False,
    )
    probability = FlatHazardRate.from_rate(ref_date, 0.02, dc365)
    discount = FlatForward.from_rate(
        ref_date, 0.03, dc365, Compounding.Continuous, Frequency.Annual,
    )
    cds = CreditDefaultSwap(
        ProtectionSide.Buyer, 10_000_000.0, 0.02, schedule, bdc, dc360,
        protection_start=ref_date, trade_date=ref_date,
    )
    return cds, probability, discount


# --- Probe schedule structure -----------------------------------------------


def test_schedule_size_and_first_date(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    """Sanity-check the schedule we build matches the C++ probe schedule."""
    cal = WeekendsOnly()
    bdc = BusinessDayConvention.Following
    maturity = ref_date + Period(5, TimeUnit.Years)
    schedule = Schedule.from_rule(
        ref_date, maturity, Period.from_frequency(Frequency.Quarterly),
        cal, bdc, bdc, DateGeneration.TwentiethIMM, False,
    )
    assert len(schedule) == cpp_ref["schedule_size"]
    assert schedule.date(0).serial_number() == cpp_ref["first_coupon_date_serial"]


# --- MidPoint engine --------------------------------------------------------


def test_midpoint_engine_npv(
    cpp_ref: dict[str, Any],
    cds_setup: tuple[CreditDefaultSwap, FlatHazardRate, FlatForward],
) -> None:
    cds, prob, disc = cds_setup
    cds.set_pricing_engine(MidPointCdsEngine(prob, 0.4, disc))
    # TIGHT: every step is closed-form discount * survival arithmetic on
    # identical inputs; bit-for-bit reproducibility is high.
    tolerance.tight(cds.npv(), cpp_ref["midpoint"]["npv"])


def test_midpoint_engine_fair_spread(
    cpp_ref: dict[str, Any],
    cds_setup: tuple[CreditDefaultSwap, FlatHazardRate, FlatForward],
) -> None:
    cds, prob, disc = cds_setup
    cds.set_pricing_engine(MidPointCdsEngine(prob, 0.4, disc))
    tolerance.tight(cds.fair_spread(), cpp_ref["midpoint"]["fair_spread"])


def test_midpoint_engine_coupon_leg_npv(
    cpp_ref: dict[str, Any],
    cds_setup: tuple[CreditDefaultSwap, FlatHazardRate, FlatForward],
) -> None:
    cds, prob, disc = cds_setup
    cds.set_pricing_engine(MidPointCdsEngine(prob, 0.4, disc))
    tolerance.tight(cds.coupon_leg_npv(), cpp_ref["midpoint"]["coupon_leg_npv"])


def test_midpoint_engine_default_leg_npv(
    cpp_ref: dict[str, Any],
    cds_setup: tuple[CreditDefaultSwap, FlatHazardRate, FlatForward],
) -> None:
    cds, prob, disc = cds_setup
    cds.set_pricing_engine(MidPointCdsEngine(prob, 0.4, disc))
    tolerance.tight(cds.default_leg_npv(), cpp_ref["midpoint"]["default_leg_npv"])


def test_midpoint_engine_coupon_leg_bps(
    cpp_ref: dict[str, Any],
    cds_setup: tuple[CreditDefaultSwap, FlatHazardRate, FlatForward],
) -> None:
    cds, prob, disc = cds_setup
    cds.set_pricing_engine(MidPointCdsEngine(prob, 0.4, disc))
    tolerance.tight(cds.coupon_leg_bps(), cpp_ref["midpoint"]["coupon_leg_bps"])


# --- Integral engine -------------------------------------------------------


def test_integral_engine_npv(
    cpp_ref: dict[str, Any],
    cds_setup: tuple[CreditDefaultSwap, FlatHazardRate, FlatForward],
) -> None:
    cds, prob, disc = cds_setup
    cds.set_pricing_engine(
        IntegralCdsEngine(Period(1, TimeUnit.Months), prob, 0.4, disc),
    )
    tolerance.tight(cds.npv(), cpp_ref["integral"]["npv"])


def test_integral_engine_fair_spread(
    cpp_ref: dict[str, Any],
    cds_setup: tuple[CreditDefaultSwap, FlatHazardRate, FlatForward],
) -> None:
    cds, prob, disc = cds_setup
    cds.set_pricing_engine(
        IntegralCdsEngine(Period(1, TimeUnit.Months), prob, 0.4, disc),
    )
    tolerance.tight(cds.fair_spread(), cpp_ref["integral"]["fair_spread"])


def test_integral_engine_coupon_leg_npv(
    cpp_ref: dict[str, Any],
    cds_setup: tuple[CreditDefaultSwap, FlatHazardRate, FlatForward],
) -> None:
    cds, prob, disc = cds_setup
    cds.set_pricing_engine(
        IntegralCdsEngine(Period(1, TimeUnit.Months), prob, 0.4, disc),
    )
    tolerance.tight(cds.coupon_leg_npv(), cpp_ref["integral"]["coupon_leg_npv"])


def test_integral_engine_default_leg_npv(
    cpp_ref: dict[str, Any],
    cds_setup: tuple[CreditDefaultSwap, FlatHazardRate, FlatForward],
) -> None:
    cds, prob, disc = cds_setup
    cds.set_pricing_engine(
        IntegralCdsEngine(Period(1, TimeUnit.Months), prob, 0.4, disc),
    )
    tolerance.tight(cds.default_leg_npv(), cpp_ref["integral"]["default_leg_npv"])


# --- Instrument inspectors -------------------------------------------------


def test_cds_inspectors(
    ref_date: Date,
    cds_setup: tuple[CreditDefaultSwap, FlatHazardRate, FlatForward],
) -> None:
    cds, _, _ = cds_setup
    assert cds.side() == ProtectionSide.Buyer
    assert cds.notional() == 10_000_000.0
    assert cds.running_spread() == 0.02
    assert cds.upfront() is None
    assert cds.settles_accrual() is True
    assert cds.pays_at_default_time() is True
    assert cds.protection_start_date() == ref_date
    assert cds.trade_date() == ref_date
    assert cds.cash_settlement_days() == 3
    assert cds.rebates_accrual() is True
    coupons = cds.coupons()
    assert len(coupons) > 0
    assert cds.upfront_payment().amount() == 0.0


def test_cds_seller_side_flips_sign(
    ref_date: Date,
    cds_setup: tuple[CreditDefaultSwap, FlatHazardRate, FlatForward],
) -> None:
    """Seller-side NPV is the negative of Buyer-side NPV (no upfront flow)."""
    cds_buyer, prob, disc = cds_setup
    cds_buyer.set_pricing_engine(MidPointCdsEngine(prob, 0.4, disc))
    npv_buyer = cds_buyer.npv()

    cal = WeekendsOnly()
    bdc = BusinessDayConvention.Following
    dc360 = Actual360()
    maturity = ref_date + Period(5, TimeUnit.Years)
    schedule = Schedule.from_rule(
        ref_date, maturity, Period.from_frequency(Frequency.Quarterly),
        cal, bdc, bdc, DateGeneration.TwentiethIMM, False,
    )
    cds_seller = CreditDefaultSwap(
        ProtectionSide.Seller, 10_000_000.0, 0.02, schedule, bdc, dc360,
        protection_start=ref_date, trade_date=ref_date,
    )
    cds_seller.set_pricing_engine(MidPointCdsEngine(prob, 0.4, disc))
    npv_seller = cds_seller.npv()
    # Sum should be 0 (or the accrual rebate, which has matching sign flips).
    tolerance.loose(npv_buyer + npv_seller, 0.0)
