"""Tests for AverageBMACoupon + BMAIndex + average_bma_leg.

Probe source: migration-harness/cpp/probes/cluster_w12c/probe.cpp
Reference:    migration-harness/references/cluster/w12c.json

The BMA coupon rate is the calendar-day-weighted average of the BMA fixings
over the interest period. With a fully-known weekly-Wednesday fixing history
the forward part never runs, so the result is deterministic. Cross-validated
against C++ v1.42.1 (099987f0). Tolerance: LOOSE (the C++ probe averages a
seeded ramp; the weighting is exact but cross-validation uses LOOSE per the
W12-C plan).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pquantlib.cashflows.average_bma_coupon import (
    AverageBMACoupon,
    average_bma_leg,
)
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.bma_index import BMAIndex
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.schedule import MakeSchedule

_TODAY = Date.from_ymd(1, Month.March, 2024)
_START = Date.from_ymd(2, Month.January, 2024)
_END = Date.from_ymd(1, Month.February, 2024)


@pytest.fixture(scope="module")
def ref() -> dict[str, float]:
    return reference_reader.load("cluster/w12c")


@pytest.fixture
def bma_with_history() -> Iterator[BMAIndex]:
    """BMA index seeded with the same deterministic ramp the probe uses."""
    idx = BMAIndex()
    idx.clear_fixings()
    ObservableSettings().evaluation_date = _TODAY
    # Seed from mid-November so that leg coupons starting in early December
    # (whose lead-in fixing rolls back ~1 week) also have a complete history.
    d = Date.from_ymd(1, Month.November, 2023)
    end_seed = Date.from_ymd(15, Month.February, 2024)
    while d <= end_seed:
        if idx.is_valid_fixing_date(d):
            idx.add_fixing(d, 0.01 + 0.0001 * (d.serial_number() % 50), force_overwrite=True)
        d = d + 1
    yield idx
    idx.clear_fixings()
    ObservableSettings().evaluation_date = None


def _payment_date(idx: BMAIndex) -> Date:
    return idx.fixing_calendar().adjust(_END, BusinessDayConvention.Following)


# --- BMAIndex --------------------------------------------------------------


def test_bma_index_name() -> None:
    assert BMAIndex().name() == "BMA1W Actual/Actual (ISDA)"


def test_bma_index_fixing_days() -> None:
    assert BMAIndex().fixing_days() == 1


def test_bma_index_valid_fixing_wednesday() -> None:
    idx = BMAIndex()
    # 3 January 2024 is a Wednesday.
    assert idx.is_valid_fixing_date(Date.from_ymd(3, Month.January, 2024))
    # 2 January 2024 is a Tuesday → not valid.
    assert not idx.is_valid_fixing_date(Date.from_ymd(2, Month.January, 2024))


# --- AverageBMACoupon ------------------------------------------------------


def test_bma_coupon_rate(bma_with_history: BMAIndex, ref: dict[str, float]) -> None:
    pay = _payment_date(bma_with_history)
    coupon = AverageBMACoupon(pay, 1_000_000.0, _START, _END, bma_with_history)
    tolerance.loose(coupon.rate(), ref["bma_rate"])


def test_bma_coupon_amount(bma_with_history: BMAIndex, ref: dict[str, float]) -> None:
    pay = _payment_date(bma_with_history)
    coupon = AverageBMACoupon(pay, 1_000_000.0, _START, _END, bma_with_history)
    tolerance.loose(coupon.amount(), ref["bma_amount"])


def test_bma_coupon_accrual_period(bma_with_history: BMAIndex, ref: dict[str, float]) -> None:
    pay = _payment_date(bma_with_history)
    coupon = AverageBMACoupon(pay, 1_000_000.0, _START, _END, bma_with_history)
    tolerance.loose(coupon.accrual_period(), ref["bma_accrual_period"])


def test_bma_coupon_fixings(bma_with_history: BMAIndex, ref: dict[str, float]) -> None:
    pay = _payment_date(bma_with_history)
    coupon = AverageBMACoupon(pay, 1_000_000.0, _START, _END, bma_with_history)
    fixings = coupon.index_fixings()
    assert len(fixings) == int(ref["bma_n_fixings"])
    tolerance.loose(fixings[0], ref["bma_first_fixing"])
    tolerance.loose(fixings[-1], ref["bma_last_fixing"])


def test_bma_coupon_single_fixing_raises(bma_with_history: BMAIndex) -> None:
    pay = _payment_date(bma_with_history)
    coupon = AverageBMACoupon(pay, 1_000_000.0, _START, _END, bma_with_history)
    with pytest.raises(LibraryException, match="no single fixing date"):
        coupon.fixing_date()
    with pytest.raises(LibraryException, match="no single fixing"):
        coupon.index_fixing()
    with pytest.raises(LibraryException, match="not defined"):
        coupon.convexity_adjustment()


def test_bma_coupon_gearing_spread(bma_with_history: BMAIndex, ref: dict[str, float]) -> None:
    pay = _payment_date(bma_with_history)
    plain = AverageBMACoupon(pay, 1_000_000.0, _START, _END, bma_with_history)
    geared = AverageBMACoupon(
        pay, 1_000_000.0, _START, _END, bma_with_history, 2.0, 0.001
    )
    # geared rate == 2 * (plain - spread=0) + 0.001  == 2*avg + 0.001
    avg = plain.rate()  # gearing 1, spread 0
    tolerance.loose(geared.rate(), 2.0 * avg + 0.001)


# --- average_bma_leg -------------------------------------------------------


def test_average_bma_leg(bma_with_history: BMAIndex) -> None:
    schedule = (
        MakeSchedule()
        .from_date(Date.from_ymd(1, Month.December, 2023))
        .to(Date.from_ymd(1, Month.February, 2024))
        .with_frequency(Frequency.Monthly)
        .with_calendar(bma_with_history.fixing_calendar())
        .with_convention(BusinessDayConvention.Following)
        .with_rule(DateGeneration.Forward)
        .build()
    )
    leg = average_bma_leg(schedule, bma_with_history, notionals=[1_000_000.0])
    assert len(leg) == 2
    for cf in leg:
        assert cf.amount() > 0.0
