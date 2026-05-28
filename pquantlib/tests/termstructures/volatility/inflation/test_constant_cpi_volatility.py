"""ConstantCPIVolatility cross-validation against C++ probe (L7-D cluster).

C++ reference values live in ``migration-harness/references/cluster/l7d.json``.

Asserts the constant surface returns the configured vol regardless of
``(date, strike)`` (Tier: EXACT — the surface is a direct quote read).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from pquantlib.daycounters.actual_actual import ActualActual
from pquantlib.daycounters.actual_actual import Convention as ActualActualConvention
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.inflation.constant_cpi_volatility import (
    ConstantCPIVolatility,
)
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF_PATH = (
    Path(__file__).resolve().parents[5] / "migration-harness/references/cluster/l7d.json"
)


@pytest.fixture(scope="module")
def cluster_refs() -> dict[str, dict[str, float]]:
    return json.loads(_REF_PATH.read_text())


@pytest.fixture(autouse=True)
def _pin_eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Pin the global evaluation date to the C++ probe's date (2024-01-17)."""
    s = ObservableSettings()
    old = s.evaluation_date
    s.evaluation_date = Date.from_ymd(17, Month.January, 2024)
    yield
    s.evaluation_date = old


def _make_surface(vol: float = 0.18) -> ConstantCPIVolatility:
    return ConstantCPIVolatility(
        vol=vol,
        settlement_days=0,
        calendar=TARGET(),
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=ActualActual(ActualActualConvention.ISDA),
        observation_lag=Period(2, TimeUnit.Months),
        frequency=Frequency.Monthly,
        index_is_interpolated=False,
    )


def test_constant_cpi_returns_configured_vol_at_any_date_strike(
    cluster_refs: dict[str, dict[str, float]],
) -> None:
    ref = cluster_refs["constant_cpi_volatility"]
    surface = _make_surface(vol=ref["vol"])
    eval_date = Date.from_ymd(17, Month.January, 2024)
    d1 = eval_date + Period(1, TimeUnit.Years)
    d5 = eval_date + Period(5, TimeUnit.Years)
    # EXACT — the surface returns the SimpleQuote value verbatim.
    exact(surface.volatility(d1, 0.03), ref["vol_at_1y_strike_3pct"])
    exact(surface.volatility(d5, 0.00), ref["vol_at_5y_strike_0pct"])


def test_constant_cpi_min_max_strike_are_infinite() -> None:
    surface = _make_surface()
    assert surface.min_strike() == float("-inf")
    assert surface.max_strike() == float("inf")


def test_constant_cpi_max_date_is_date_max() -> None:
    surface = _make_surface()
    assert surface.max_date() == Date.max_date()


def test_constant_cpi_accepts_quote_input() -> None:
    quote = SimpleQuote(0.42)
    surface = ConstantCPIVolatility(
        vol=quote,
        settlement_days=0,
        calendar=TARGET(),
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        day_counter=ActualActual(ActualActualConvention.ISDA),
        observation_lag=Period(2, TimeUnit.Months),
        frequency=Frequency.Monthly,
        index_is_interpolated=False,
    )
    eval_date = Date.from_ymd(17, Month.January, 2024)
    tight(surface.volatility(eval_date + Period(1, TimeUnit.Years), 0.03), 0.42)
    quote.set_value(0.20)
    # Should reflect the updated quote.
    tight(surface.volatility(eval_date + Period(1, TimeUnit.Years), 0.03), 0.20)


def test_constant_cpi_inspectors_round_trip() -> None:
    surface = _make_surface()
    assert surface.observation_lag() == Period(2, TimeUnit.Months)
    assert surface.frequency() == Frequency.Monthly
    assert surface.index_is_interpolated() is False
