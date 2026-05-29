"""Smoke tests for PiecewiseDefaultCurve scaffold.

# C++ parity: ql/termstructures/credit/piecewisedefaultcurve.hpp.

Bootstrap is deferred — these tests verify the constructor wiring + the
``not-yet-implemented`` raises for the rate accessors.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.credit.default_probability_helpers import SpreadCdsHelper
from pquantlib.termstructures.credit.piecewise_default_curve import (
    PiecewiseDefaultCurve,
)
from pquantlib.termstructures.credit.probability_traits import HazardRateTrait
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
def helpers_and_curve() -> Iterator[tuple[Date, list[SpreadCdsHelper]]]:
    d = Date.from_ymd(15, Month.June, 2026)
    ObservableSettings().evaluation_date = d
    disc = FlatForward.from_rate(
        d, 0.03, Actual365Fixed(),
        Compounding.Continuous, Frequency.Annual,
    )
    helpers = [
        SpreadCdsHelper(
            quote=0.02,
            tenor=Period(t, TimeUnit.Years),
            settlement_days=1,
            calendar=WeekendsOnly(),
            frequency=Frequency.Quarterly,
            payment_convention=BusinessDayConvention.Following,
            rule=DateGeneration.Backward,
            day_counter=Actual360(),
            recovery_rate=0.4,
            discount_curve=disc,
        )
        for t in (1, 3, 5)
    ]
    yield d, helpers
    ObservableSettings().evaluation_date = None


def test_piecewise_default_curve_constructs(
    helpers_and_curve: tuple[Date, list[SpreadCdsHelper]],
) -> None:
    d, helpers = helpers_and_curve
    curve = PiecewiseDefaultCurve(HazardRateTrait, d, helpers, Actual365Fixed())
    assert curve.traits() is HazardRateTrait
    assert len(curve.instruments()) == 3
    # max_date is the latest helper date.
    assert curve.max_date() == max(h.latest_date() for h in helpers)


def test_piecewise_default_curve_requires_at_least_one_helper(
    helpers_and_curve: tuple[Date, list[SpreadCdsHelper]],
) -> None:
    d, _ = helpers_and_curve
    with pytest.raises(LibraryException, match="at least one instrument"):
        PiecewiseDefaultCurve(HazardRateTrait, d, [], Actual365Fixed())


def test_piecewise_default_curve_survival_probability_deferred(
    helpers_and_curve: tuple[Date, list[SpreadCdsHelper]],
) -> None:
    d, helpers = helpers_and_curve
    curve = PiecewiseDefaultCurve(HazardRateTrait, d, helpers, Actual365Fixed())
    with pytest.raises(LibraryException, match="bootstrap is deferred"):
        curve.survival_probability(1.0, extrapolate=True)


def test_piecewise_default_curve_default_density_deferred(
    helpers_and_curve: tuple[Date, list[SpreadCdsHelper]],
) -> None:
    d, helpers = helpers_and_curve
    curve = PiecewiseDefaultCurve(HazardRateTrait, d, helpers, Actual365Fixed())
    with pytest.raises(LibraryException, match="bootstrap is deferred"):
        curve.default_density(1.0, extrapolate=True)
