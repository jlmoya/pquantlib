"""Tests for SwaptionVolatilityDiscrete (via its SwaptionVolatilityMatrix subclass)."""

from __future__ import annotations

import numpy as np

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.termstructures.volatility.swaption.swaption_volatility_matrix import (
    SwaptionVolatilityMatrix,
)
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def _new() -> SwaptionVolatilityMatrix:
    return SwaptionVolatilityMatrix(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=[
            Period(1, TimeUnit.Years),
            Period(2, TimeUnit.Years),
            Period(5, TimeUnit.Years),
        ],
        swap_tenors=[Period(1, TimeUnit.Years), Period(5, TimeUnit.Years)],
        volatilities=np.full((3, 2), 0.20),
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=Date.from_ymd(15, Month.January, 2024),
    )


def test_pillar_count_matches_input() -> None:
    m = _new()
    assert len(m.option_tenors()) == 3
    assert len(m.swap_tenors()) == 2
    assert len(m.option_dates()) == 3
    assert len(m.option_times()) == 3
    assert len(m.swap_lengths()) == 2


def test_option_dates_increase() -> None:
    m = _new()
    dates = m.option_dates()
    for i in range(1, len(dates)):
        assert dates[i] > dates[i - 1]


def test_option_times_increase() -> None:
    m = _new()
    times = m.option_times()
    for i in range(1, len(times)):
        assert times[i] > times[i - 1]


def test_swap_lengths_are_year_fractions() -> None:
    m = _new()
    lengths = m.swap_lengths()
    tolerance.tight(lengths[0], 1.0)
    tolerance.tight(lengths[1], 5.0)
