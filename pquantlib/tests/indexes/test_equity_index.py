"""Cross-validate ``EquityIndex`` against C++ QuantLib v1.43.

Probe: ``v143/cf/equitycashflow`` (section ``equity_index``).

The market is the one used by the C++ test suite (test-suite/equityindex.cpp):
TARGET / Actual365Fixed, evaluation date 27 January 2023, flat 3.75% interest,
flat 0.5% dividend, spot 8700 and two historical fixings.

What matters here is that all three branches of ``forecast_fixing`` are
exercised with values that differ from one another, and that each optional
constructor argument (``interest`` / ``dividend`` / ``spot``) has a test at a
non-default value whose consequence is visible in a number.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.equity_index import EquityIndex
from pquantlib.indexes.index_manager import IndexManager
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_TODAY = Date.from_ymd(27, Month.January, 2023)
_BASE_DATE = Date.from_ymd(5, Month.January, 2023)
_NEW_YEARS_DAY = Date.from_ymd(1, Month.January, 2023)
_SATURDAY = Date.from_ymd(28, Month.January, 2023)
_NO_FIXING_DAY = Date.from_ymd(2, Month.January, 2023)

_BASE_FIXING = 9010.0
_TODAYS_FIXING = 8690.0
_SPOT = 8700.0
_INTEREST_RATE = 0.0375
_DIVIDEND_RATE = 0.005


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/cf/equitycashflow")["equity_index"]


@pytest.fixture(autouse=True)
def _global_state() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    previous = ObservableSettings().evaluation_date
    ObservableSettings().evaluation_date = _TODAY
    IndexManager().clear_histories()
    yield
    IndexManager().clear_histories()
    ObservableSettings().evaluation_date = previous


class _Market:
    """The C++ ``CommonVars`` fixture, rebuilt per test."""

    def __init__(self, *, add_todays_fixing: bool = True) -> None:
        self.calendar = TARGET()
        self.day_count = Actual365Fixed()
        self.interest = FlatForward.from_rate(_TODAY, _INTEREST_RATE, self.day_count)
        self.dividend = FlatForward.from_rate(_TODAY, _DIVIDEND_RATE, self.day_count)
        self.spot = SimpleQuote(_SPOT)
        self.index = EquityIndex(
            "eqIndex", self.calendar, EURCurrency(), self.interest, self.dividend, self.spot
        )
        self.index.add_fixing(_BASE_DATE, _BASE_FIXING)
        if add_todays_fixing:
            self.index.add_fixing(_TODAY, _TODAYS_FIXING)


def _check_forecast_table(index: EquityIndex, ref: dict[str, Any]) -> None:
    """``fixing`` and ``forecast_fixing`` at the three pinned future dates."""
    for i in (1, 2, 3):
        d = Date(int(ref[f"date_serial_{i}"]))
        tolerance.tight(index.fixing(d), ref[f"fixing_{i}"])
        tolerance.tight(index.forecast_fixing(d), ref[f"forecast_fixing_{i}"])


# --- wiring -------------------------------------------------------------------


def test_wiring(cpp: dict[str, Any]) -> None:
    index = _Market().index
    assert index.name() == cpp["name"]
    assert index.fixing_calendar().name() == cpp["fixing_calendar"]
    assert index.currency().code == cpp["currency_code"]


def test_is_valid_fixing_date(cpp: dict[str, Any]) -> None:
    index = _Market().index
    assert _TODAY.serial_number() == cpp["valid_business_day_serial"]
    assert _NEW_YEARS_DAY.serial_number() == cpp["holiday_serial"]
    assert _SATURDAY.serial_number() == cpp["weekend_serial"]
    assert index.is_valid_fixing_date(_TODAY) is cpp["is_valid_business_day"]
    assert index.is_valid_fixing_date(_NEW_YEARS_DAY) is cpp["is_valid_holiday"]
    assert index.is_valid_fixing_date(_SATURDAY) is cpp["is_valid_weekend"]


def test_default_optional_arguments_are_empty() -> None:
    """The three optional handles default to "empty" (``None``)."""
    index = EquityIndex("bareIndex", TARGET(), EURCurrency())
    assert index.equity_interest_rate_curve() is None
    assert index.equity_dividend_curve() is None
    assert index.spot() is None


# --- past fixings -------------------------------------------------------------


def test_past_fixing_round_trip(cpp: dict[str, Any]) -> None:
    index = _Market().index
    assert index.has_historical_fixing(_BASE_DATE) is cpp["has_historical_fixing_base"]
    tolerance.exact(index.past_fixing(_BASE_DATE), cpp["past_fixing_base"])
    tolerance.exact(index.fixing(_BASE_DATE), cpp["fixing_base_date"])
    tolerance.exact(index.past_fixing(_TODAY), cpp["past_fixing_today"])
    tolerance.exact(index.fixing(_TODAY), cpp["fixing_today"])


def test_forecast_todays_fixing_flag_overrides_history(cpp: dict[str, Any]) -> None:
    """``forecast_todays_fixing=True`` (non-default) must reach the spot quote.

    History says 8690, the spot quote says 8700: the flag is the only thing
    that distinguishes them.
    """
    index = _Market().index
    tolerance.exact(index.fixing(_TODAY, False), cpp["fixing_today"])
    tolerance.tight(index.fixing(_TODAY, True), cpp["fixing_today_forecast"])
    assert cpp["fixing_today"] != cpp["fixing_today_forecast"]


def test_spot_proxies_a_missing_todays_fixing(cpp: dict[str, Any]) -> None:
    market = _Market(add_todays_fixing=False)
    tolerance.tight(market.index.fixing(_TODAY), cpp["spot_proxy_today"])


# --- forecasting: the three branches -----------------------------------------


def test_forecast_with_dividend_curve(cpp: dict[str, Any]) -> None:
    market = _Market()
    assert market.index.equity_dividend_curve() is market.dividend
    assert market.index.equity_interest_rate_curve() is market.interest
    _check_forecast_table(market.index, cpp["forecast_with_dividend"])


def test_forecast_without_dividend_curve(cpp: dict[str, Any]) -> None:
    """Empty dividend handle: the interest curve alone is the forward curve."""
    market = _Market()
    ex_dividend = market.index.clone(market.interest, None, market.spot)
    assert ex_dividend.equity_dividend_curve() is None
    _check_forecast_table(ex_dividend, cpp["forecast_no_dividend"])
    # The dividend argument is not decorative: dropping it moves every number.
    assert cpp["forecast_no_dividend"]["fixing_1"] != cpp["forecast_with_dividend"]["fixing_1"]


def test_forecast_without_spot_uses_last_historical_fixing(cpp: dict[str, Any]) -> None:
    market = _Market()
    ex_spot = market.index.clone(market.interest, market.dividend, None)
    assert ex_spot.spot() is None
    _check_forecast_table(ex_spot, cpp["forecast_no_spot"])
    assert cpp["forecast_no_spot"]["fixing_1"] != cpp["forecast_with_dividend"]["fixing_1"]


def test_clone_relinked_uses_the_new_curves(cpp: dict[str, Any]) -> None:
    market = _Market()
    day_count = market.day_count
    clone = market.index.clone(
        FlatForward.from_rate(_TODAY, 0.06, day_count),
        FlatForward.from_rate(_TODAY, 0.02, day_count),
        SimpleQuote(9000.0),
    )
    assert clone.name() == cpp["clone_name"]
    assert clone.fixing_calendar().name() == cpp["clone_calendar"]
    assert clone.currency().code == cpp["clone_currency_code"]
    spot = clone.spot()
    assert spot is not None
    tolerance.exact(spot.value(), cpp["clone_spot_value"])
    _check_forecast_table(clone, cpp["forecast_clone_relinked"])
    # The clone must actually be using the relinked curves, not the originals.
    assert cpp["forecast_clone_relinked"]["fixing_3"] != cpp["forecast_with_dividend"]["fixing_3"]


# --- observability ------------------------------------------------------------


class _Flag:
    """Minimal Observer that records that it was notified."""

    def __init__(self) -> None:
        self.up = False

    def update(self) -> None:
        self.up = True


def test_spot_quote_change_propagates_through_the_index() -> None:
    """The ``spot`` argument is registered with, not just stored."""
    market = _Market()
    flag = _Flag()
    market.index.register_with(flag)
    assert flag.up is False
    market.spot.set_value(9000.0)
    assert flag.up is True


def test_interest_curve_change_propagates_through_the_index() -> None:
    """Same for the ``interest`` argument."""
    market = _Market()
    rate = SimpleQuote(_INTEREST_RATE)
    index = EquityIndex(
        "eqIndexObservable",
        market.calendar,
        EURCurrency(),
        FlatForward(_TODAY, rate, market.day_count),
        market.dividend,
        market.spot,
    )
    flag = _Flag()
    index.register_with(flag)
    assert flag.up is False
    rate.set_value(0.05)
    assert flag.up is True


# --- failure branches ---------------------------------------------------------


def test_raises_on_invalid_fixing_date(cpp: dict[str, Any]) -> None:
    assert cpp["raises"]["fixing_on_holiday"] == {"raises": True}
    index = _Market().index
    with pytest.raises(LibraryException, match="is not valid"):
        index.fixing(_NEW_YEARS_DAY)


def test_raises_when_past_fixing_missing(cpp: dict[str, Any]) -> None:
    assert cpp["raises"]["fixing_missing_past"] == {"raises": True}
    index = _Market().index
    with pytest.raises(LibraryException, match="Missing eqIndex fixing"):
        index.fixing(_NO_FIXING_DAY)


def test_raises_when_interest_curve_missing(cpp: dict[str, Any]) -> None:
    assert cpp["raises"]["forecast_without_interest_curve"] == {"raises": True}
    market = _Market()
    bare = market.index.clone(None, None, None)
    future = Date(int(cpp["forecast_with_dividend"]["date_serial_3"]))
    with pytest.raises(LibraryException, match="null interest rate term structure"):
        bare.fixing(future)


def test_raises_without_spot_and_without_history(cpp: dict[str, Any]) -> None:
    assert cpp["raises"]["forecast_without_spot_and_history"] == {"raises": True}
    market = _Market(add_todays_fixing=False)
    IndexManager().clear_histories()
    ex_spot = market.index.clone(market.interest, market.dividend, None)
    future = Date(int(cpp["forecast_with_dividend"]["date_serial_3"]))
    with pytest.raises(LibraryException, match="missing both spot and historical index"):
        ex_spot.fixing(future)
