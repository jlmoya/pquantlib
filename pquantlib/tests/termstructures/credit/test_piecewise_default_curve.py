"""Tests for PiecewiseDefaultCurve — wired-up bootstrap.

# C++ parity: ql/termstructures/credit/piecewisedefaultcurve.hpp.

L9-B wires the L8-B scaffold with a full
``IterativeBootstrap[DefaultProbabilityTermStructure, Traits]`` (from
L8-A). Verifies:
- constructor wiring + helper count + max_date.
- HazardRate traits: 3-CDS roundtrip — implied_quote ≈ input quote
  to LOOSE tolerance.
- SurvivalProbability traits: same shape, log-linear underlying.
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
from pquantlib.termstructures.credit.probability_traits import (
    HazardRateTrait,
    SurvivalProbabilityTrait,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.tolerance import loose
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


def test_piecewise_default_curve_requires_at_least_one_helper(
    helpers_and_curve: tuple[Date, list[SpreadCdsHelper]],
) -> None:
    d, _ = helpers_and_curve
    with pytest.raises(LibraryException, match="at least one instrument"):
        PiecewiseDefaultCurve(HazardRateTrait, d, [], Actual365Fixed())


def test_piecewise_default_curve_max_date_matches_helpers_before_bootstrap(
    helpers_and_curve: tuple[Date, list[SpreadCdsHelper]],
) -> None:
    d, helpers = helpers_and_curve
    curve = PiecewiseDefaultCurve(HazardRateTrait, d, helpers, Actual365Fixed())
    # Before bootstrap runs, max_date falls back to the latest helper.
    assert curve.max_date() == max(h.latest_date() for h in helpers)


def test_piecewise_default_curve_hazard_rate_traits_roundtrip(
    helpers_and_curve: tuple[Date, list[SpreadCdsHelper]],
) -> None:
    """Bootstrapped curve reproduces input CDS quotes (HazardRate)."""
    d, helpers = helpers_and_curve
    curve = PiecewiseDefaultCurve(HazardRateTrait, d, helpers, Actual365Fixed())
    # Trigger bootstrap.
    _ = curve.survival_probability(1.0, extrapolate=True)
    # All helpers should be re-priced to their input quote (0.02) by
    # the bootstrapped curve to LOOSE tolerance.
    for h in helpers:
        implied = h.implied_quote()
        loose(implied, 0.02)


def test_piecewise_default_curve_survival_probability_traits_roundtrip(
    helpers_and_curve: tuple[Date, list[SpreadCdsHelper]],
) -> None:
    """Bootstrapped curve reproduces input CDS quotes (SurvivalProbability)."""
    d, helpers = helpers_and_curve
    curve = PiecewiseDefaultCurve(SurvivalProbabilityTrait, d, helpers, Actual365Fixed())
    _ = curve.survival_probability(1.0, extrapolate=True)
    for h in helpers:
        implied = h.implied_quote()
        loose(implied, 0.02)


def test_piecewise_default_curve_data_after_bootstrap(
    helpers_and_curve: tuple[Date, list[SpreadCdsHelper]],
) -> None:
    """After bootstrap the curve exposes a non-trivial data grid."""
    d, helpers = helpers_and_curve
    curve = PiecewiseDefaultCurve(HazardRateTrait, d, helpers, Actual365Fixed())
    _ = curve.survival_probability(1.0, extrapolate=True)
    assert len(curve.data()) == len(helpers) + 1
    # Hazard rates should be positive.
    assert all(x > 0 for x in curve.data())
