"""Tests for PiecewiseYieldCurve — concrete bootstrap yield curve.

# C++ parity: ql/termstructures/yield/piecewiseyieldcurve.hpp.

L9-B closes the L2-B carve-out: PiecewiseYieldCurve runs the full
``IterativeBootstrap[YieldTermStructure, Traits]`` loop from L8-A.

Tests cover the three traits (Discount / ZeroYield / ForwardRate) and
the roundtrip property: bootstrapped curve reproduces input quotes to
LOOSE tolerance.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.deposit_rate_helper import DepositRateHelper
from pquantlib.termstructures.yield_.piecewise_yield_curve import PiecewiseYieldCurve
from pquantlib.termstructures.yield_.yield_traits import (
    Discount,
    ForwardRate,
    ZeroYield,
)
from pquantlib.testing.tolerance import loose
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture
def deposit_helpers() -> Iterator[tuple[Date, list[DepositRateHelper]]]:
    """3 deposit helpers at 1M / 3M / 6M with rates 0.02 / 0.025 / 0.03."""
    d = Date.from_ymd(15, Month.January, 2024)
    ObservableSettings().evaluation_date = d
    helpers = [
        DepositRateHelper(
            SimpleQuote(rate),
            tenor=Period(months, TimeUnit.Months),
            fixing_days=2,
            calendar=TARGET(),
            convention=BusinessDayConvention.ModifiedFollowing,
            end_of_month=True,
            day_counter=Actual360(),
            evaluation_date=d,
        )
        for months, rate in [(1, 0.02), (3, 0.025), (6, 0.03)]
    ]
    yield d, helpers
    ObservableSettings().evaluation_date = None


def test_piecewise_yield_curve_constructs(
    deposit_helpers: tuple[Date, list[DepositRateHelper]],
) -> None:
    d, helpers = deposit_helpers
    curve = PiecewiseYieldCurve(Discount, d, helpers, Actual360())
    assert curve.traits() is Discount
    assert len(curve.instruments()) == 3


def test_piecewise_yield_curve_requires_at_least_one_helper(
    deposit_helpers: tuple[Date, list[DepositRateHelper]],
) -> None:
    d, _ = deposit_helpers
    with pytest.raises(LibraryException, match="at least one instrument"):
        PiecewiseYieldCurve(Discount, d, [], Actual360())


def test_piecewise_yield_curve_discount_traits_roundtrip(
    deposit_helpers: tuple[Date, list[DepositRateHelper]],
) -> None:
    """Bootstrapped curve reproduces input deposit quotes (Discount traits)."""
    d, helpers = deposit_helpers
    curve = PiecewiseYieldCurve(Discount, d, helpers, Actual360())
    # Trigger bootstrap.
    _ = curve.discount(0.5)
    # All helpers should re-price to their input quote.
    expected_rates = [0.02, 0.025, 0.03]
    for h, expected in zip(helpers, expected_rates, strict=True):
        implied = h.implied_quote()
        loose(implied, expected)


def test_piecewise_yield_curve_zero_yield_traits_roundtrip(
    deposit_helpers: tuple[Date, list[DepositRateHelper]],
) -> None:
    """Bootstrapped curve reproduces input deposit quotes (ZeroYield traits)."""
    d, helpers = deposit_helpers
    curve = PiecewiseYieldCurve(ZeroYield, d, helpers, Actual360())
    _ = curve.discount(0.5)
    expected_rates = [0.02, 0.025, 0.03]
    for h, expected in zip(helpers, expected_rates, strict=True):
        implied = h.implied_quote()
        loose(implied, expected)


def test_piecewise_yield_curve_forward_rate_traits_roundtrip(
    deposit_helpers: tuple[Date, list[DepositRateHelper]],
) -> None:
    """Bootstrapped curve reproduces input deposit quotes (ForwardRate traits)."""
    d, helpers = deposit_helpers
    curve = PiecewiseYieldCurve(ForwardRate, d, helpers, Actual360())
    _ = curve.discount(0.5)
    expected_rates = [0.02, 0.025, 0.03]
    for h, expected in zip(helpers, expected_rates, strict=True):
        implied = h.implied_quote()
        loose(implied, expected)


def test_piecewise_yield_curve_discount_factor_positive(
    deposit_helpers: tuple[Date, list[DepositRateHelper]],
) -> None:
    """Discount factors must be in (0, 1] for a standard upward curve."""
    d, helpers = deposit_helpers
    curve = PiecewiseYieldCurve(Discount, d, helpers, Actual360())
    df_3m = curve.discount(0.25)
    df_6m = curve.discount(0.5)
    assert 0.0 < df_6m < df_3m <= 1.0


def test_piecewise_yield_curve_data_after_bootstrap(
    deposit_helpers: tuple[Date, list[DepositRateHelper]],
) -> None:
    """After bootstrap the curve exposes n+1 nodes."""
    d, helpers = deposit_helpers
    curve = PiecewiseYieldCurve(Discount, d, helpers, Actual360())
    _ = curve.discount(0.5)
    assert len(curve.data()) == len(helpers) + 1
    assert curve.data()[0] == 1.0  # discount(0) = 1
