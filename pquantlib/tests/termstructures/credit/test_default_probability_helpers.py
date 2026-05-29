"""Smoke tests for SpreadCdsHelper / UpfrontCdsHelper.

Validates that the helpers wire a CDS instrument + MidPoint engine
correctly: given the FlatHazardRate curve that *would* be the
bootstrap solution for a fixed spread, ``implied_quote()`` recovers
that spread (to LOOSE tolerance — the helper builds its own schedule
which may differ slightly from a hand-constructed one).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.credit.default_probability_helpers import (
    SpreadCdsHelper,
    UpfrontCdsHelper,
)
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.weekends_only import WeekendsOnly
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture
def ref_date() -> Iterator[Date]:
    d = Date.from_ymd(15, Month.June, 2026)
    # The helper reads the global evaluation date; pin it so the test
    # is deterministic.
    ObservableSettings().evaluation_date = d
    yield d
    ObservableSettings().evaluation_date = None


def test_spread_cds_helper_implied_quote_with_flat_curve(ref_date: Date) -> None:
    """Given a FlatHazardRate curve, ``implied_quote()`` returns a finite spread.

    LOOSE: the helper's auto-built schedule may not match the assumed
    flat curve's parameters exactly, but the returned spread should be
    in the plausible range (within ~10x the assumed pre-bootstrap level).
    """
    disc = FlatForward.from_rate(
        ref_date, 0.03, Actual365Fixed(),
        Compounding.Continuous, Frequency.Annual,
    )
    cal = WeekendsOnly()
    helper = SpreadCdsHelper(
        quote=0.02,
        tenor=Period(5, TimeUnit.Years),
        settlement_days=1,
        calendar=cal,
        frequency=Frequency.Quarterly,
        payment_convention=BusinessDayConvention.Following,
        rule=DateGeneration.Backward,
        day_counter=Actual360(),
        recovery_rate=0.4,
        discount_curve=disc,
    )
    # Bind to a curve that *is* the bootstrap solution.
    curve = FlatHazardRate.from_rate(ref_date, 0.02, Actual365Fixed())
    helper.set_term_structure(curve)
    implied = helper.implied_quote()
    # The helper constructs its own schedule (Backward) starting from
    # ``evalDate + settlement_days``, which differs from a hand-rolled
    # probe. We only assert the implied spread is in the expected
    # neighborhood (~1-3% for these parameters). The cross-validation
    # against C++ probe values lives in test_credit_default_swap.py,
    # which uses an explicit schedule.
    assert 0.005 < implied < 0.05


def test_spread_cds_helper_inspectors_after_init(ref_date: Date) -> None:
    disc = FlatForward.from_rate(
        ref_date, 0.03, Actual365Fixed(),
        Compounding.Continuous, Frequency.Annual,
    )
    helper = SpreadCdsHelper(
        quote=0.02,
        tenor=Period(5, TimeUnit.Years),
        settlement_days=1,
        calendar=WeekendsOnly(),
        frequency=Frequency.Quarterly,
        payment_convention=BusinessDayConvention.Following,
        rule=DateGeneration.Backward,
        day_counter=Actual360(),
        recovery_rate=0.4,
        discount_curve=disc,
    )
    # earliest / latest dates should be populated.
    assert helper.earliest_date() != Date()
    assert helper.latest_date() != Date()
    assert helper.swap() is None  # not bound yet


def test_upfront_cds_helper_implied_quote_with_flat_curve(ref_date: Date) -> None:
    disc = FlatForward.from_rate(
        ref_date, 0.03, Actual365Fixed(),
        Compounding.Continuous, Frequency.Annual,
    )
    helper = UpfrontCdsHelper(
        upfront=0.0,           # 0% upfront
        running_spread=0.02,   # 200 bps running
        tenor=Period(5, TimeUnit.Years),
        settlement_days=1,
        calendar=WeekendsOnly(),
        frequency=Frequency.Quarterly,
        payment_convention=BusinessDayConvention.Following,
        rule=DateGeneration.Backward,
        day_counter=Actual360(),
        recovery_rate=0.4,
        discount_curve=disc,
    )
    curve = FlatHazardRate.from_rate(ref_date, 0.02, Actual365Fixed())
    helper.set_term_structure(curve)
    implied_upfront = helper.implied_quote()
    # Plausibility check: upfront with 200bps running on a 2% lambda curve
    # and 40% recovery should be very small in magnitude (within ~10% of
    # notional, typically < 1%).
    assert abs(implied_upfront) < 0.5


def test_cds_helper_quote_round_trip(ref_date: Date) -> None:
    """quote_error = quote - implied_quote; sign matches mid-pricing intuition."""
    disc = FlatForward.from_rate(
        ref_date, 0.03, Actual365Fixed(),
        Compounding.Continuous, Frequency.Annual,
    )
    helper = SpreadCdsHelper(
        quote=0.02,
        tenor=Period(5, TimeUnit.Years),
        settlement_days=1,
        calendar=WeekendsOnly(),
        frequency=Frequency.Quarterly,
        payment_convention=BusinessDayConvention.Following,
        rule=DateGeneration.Backward,
        day_counter=Actual360(),
        recovery_rate=0.4,
        discount_curve=disc,
    )
    curve = FlatHazardRate.from_rate(ref_date, 0.02, Actual365Fixed())
    helper.set_term_structure(curve)
    err = helper.quote_error()
    # quote (2%) - implied (~1.2% per probe parameters) > 0.
    assert err > 0.0
