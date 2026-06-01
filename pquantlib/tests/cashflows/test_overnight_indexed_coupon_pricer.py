"""Tests for the full-fidelity overnight-indexed coupon pricers.

Probe source: migration-harness/cpp/probes/cluster_w12c/probe.cpp
Reference:    migration-harness/references/cluster/w12c.json

Covers CompoundingOvernightIndexedCouponPricer and
ArithmeticAveragedOvernightIndexedCouponPricer over a fully-known overnight
(Sofr) fixing history — deterministic compounding / averaging, no forward
curve. Cross-validated TIGHT against C++ v1.42.1 (099987f0).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pquantlib.cashflows.coupon_pricer import CouponPricer
from pquantlib.cashflows.ibor_coupon import IborCoupon
from pquantlib.cashflows.overnight_indexed_coupon import OvernightIndexedCoupon
from pquantlib.cashflows.overnight_indexed_coupon_pricer import (
    ArithmeticAveragedOvernightIndexedCouponPricer,
    CompoundingOvernightIndexedCouponPricer,
    OvernightIndexedCouponPricer,
)
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.ibor.sofr import Sofr
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_TODAY = Date.from_ymd(1, Month.March, 2024)
_START = Date.from_ymd(1, Month.February, 2024)
_END = Date.from_ymd(1, Month.March, 2024)


@pytest.fixture(scope="module")
def ref() -> dict[str, float]:
    return reference_reader.load("cluster/w12c")


@pytest.fixture
def sofr_with_history() -> Iterator[Sofr]:
    """Sofr index seeded with a deterministic ramp of past fixings.

    Evaluation date pinned to 1-Mar-2024 (after the coupon window), so every
    fixing is historical and the deterministic compute path runs.
    """
    idx = Sofr()
    idx.clear_fixings()
    ObservableSettings().evaluation_date = _TODAY
    # Seed fixings on the coupon's fixing dates with the same ramp the probe
    # uses: 0.05 + 0.0001 * i.
    cal = idx.fixing_calendar()
    pay = cal.adjust(_END, BusinessDayConvention.Following)
    template = OvernightIndexedCoupon(pay, 1_000_000.0, _START, _END, idx)
    for i, d in enumerate(template.fixing_dates()):
        idx.add_fixing(d, 0.05 + 0.0001 * i, force_overwrite=True)
    yield idx
    idx.clear_fixings()
    ObservableSettings().evaluation_date = None


def _payment_date(idx: Sofr) -> Date:
    return idx.fixing_calendar().adjust(_END, BusinessDayConvention.Following)


# --- Compounding -----------------------------------------------------------


def test_compounding_rate(sofr_with_history: Sofr, ref: dict[str, float]) -> None:
    pay = _payment_date(sofr_with_history)
    coupon = OvernightIndexedCoupon(pay, 1_000_000.0, _START, _END, sofr_with_history)
    coupon.set_pricer(CompoundingOvernightIndexedCouponPricer())
    tolerance.tight(coupon.rate(), ref["on_compound_rate"])


def test_compounding_amount(sofr_with_history: Sofr, ref: dict[str, float]) -> None:
    pay = _payment_date(sofr_with_history)
    coupon = OvernightIndexedCoupon(pay, 1_000_000.0, _START, _END, sofr_with_history)
    coupon.set_pricer(CompoundingOvernightIndexedCouponPricer())
    tolerance.tight(coupon.amount(), ref["on_compound_amount"])


def test_compounding_n_fixings(sofr_with_history: Sofr, ref: dict[str, float]) -> None:
    pay = _payment_date(sofr_with_history)
    coupon = OvernightIndexedCoupon(pay, 1_000_000.0, _START, _END, sofr_with_history)
    assert len(coupon.fixing_dates()) == int(ref["on_n_fixings"])


def test_compounding_gearing_spread(sofr_with_history: Sofr, ref: dict[str, float]) -> None:
    pay = _payment_date(sofr_with_history)
    coupon = OvernightIndexedCoupon(
        pay, 1_000_000.0, _START, _END, sofr_with_history, 2.0, 0.001
    )
    coupon.set_pricer(CompoundingOvernightIndexedCouponPricer())
    # gearing 2 * avg + spread 0.001
    tolerance.tight(coupon.rate(), ref["on_compound_gs_rate"])


def test_compounding_effective_spread_and_fixing(sofr_with_history: Sofr) -> None:
    """With spread not compounded daily, effectiveSpread == spread."""
    pay = _payment_date(sofr_with_history)
    coupon = OvernightIndexedCoupon(
        pay, 1_000_000.0, _START, _END, sofr_with_history, 1.0, 0.0025
    )
    pricer = CompoundingOvernightIndexedCouponPricer()
    coupon.set_pricer(pricer)
    pricer.initialize(coupon)
    tolerance.tight(pricer.effective_spread(), 0.0025)
    # swaplet = gearing*effectiveIndexFixing + effectiveSpread
    swaplet = coupon.gearing() * pricer.effective_index_fixing() + pricer.effective_spread()
    tolerance.tight(coupon.rate(), swaplet)


# --- Arithmetic average ----------------------------------------------------


def test_arithmetic_rate(sofr_with_history: Sofr, ref: dict[str, float]) -> None:
    pay = _payment_date(sofr_with_history)
    coupon = OvernightIndexedCoupon(pay, 1_000_000.0, _START, _END, sofr_with_history)
    coupon.set_pricer(ArithmeticAveragedOvernightIndexedCouponPricer())
    tolerance.tight(coupon.rate(), ref["on_arith_rate"])


def test_arithmetic_amount(sofr_with_history: Sofr, ref: dict[str, float]) -> None:
    pay = _payment_date(sofr_with_history)
    coupon = OvernightIndexedCoupon(pay, 1_000_000.0, _START, _END, sofr_with_history)
    coupon.set_pricer(ArithmeticAveragedOvernightIndexedCouponPricer())
    tolerance.tight(coupon.amount(), ref["on_arith_amount"])


def test_arithmetic_below_compounding(sofr_with_history: Sofr) -> None:
    """Arithmetic average omits cross-product terms, so it is < compounded."""
    pay = _payment_date(sofr_with_history)
    cc = OvernightIndexedCoupon(pay, 1_000_000.0, _START, _END, sofr_with_history)
    cc.set_pricer(CompoundingOvernightIndexedCouponPricer())
    ca = OvernightIndexedCoupon(pay, 1_000_000.0, _START, _END, sofr_with_history)
    ca.set_pricer(ArithmeticAveragedOvernightIndexedCouponPricer())
    assert ca.rate() < cc.rate()


# --- type / interface ------------------------------------------------------


def test_pricers_are_coupon_pricers() -> None:
    assert isinstance(CompoundingOvernightIndexedCouponPricer(), CouponPricer)
    assert isinstance(ArithmeticAveragedOvernightIndexedCouponPricer(), CouponPricer)
    assert isinstance(
        CompoundingOvernightIndexedCouponPricer(), OvernightIndexedCouponPricer
    )


def test_base_pricer_rejects_non_overnight_coupon() -> None:
    ibor = Euribor(Period(6, TimeUnit.Months))
    ic = IborCoupon(
        Date.from_ymd(1, Month.August, 2024),
        100_000.0,
        Date.from_ymd(1, Month.February, 2024),
        Date.from_ymd(1, Month.August, 2024),
        0,
        ibor,
    )
    pricer = CompoundingOvernightIndexedCouponPricer()
    with pytest.raises(LibraryException, match="unsupported coupon type"):
        pricer.initialize(ic)
