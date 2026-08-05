"""Cross-validate EquityIndex + the batch-fixing semantics against C++ v1.43.

Probe: ``v143/indexes/equity``.

EquityIndex is the one index in the v1.43 coverage batch with a real forecast
rather than pure wiring, so all three forward paths are pinned (with dividend
curve, without, and with the spot resolved from history), together with the
branch selection inside ``fixing()``.

``Index.add_fixings`` is pinned for its ordering guarantee rather than its
happy path: C++ stores every acceptable fixing and only then raises for the
rejected ones, so a batch containing one invalid date still commits the valid
entries.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.equity_index import EquityIndex
from pquantlib.indexes.ibor.euribor import Euribor3M
from pquantlib.indexes.index_manager import IndexManager
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.united_states import UnitedStates
from pquantlib.time.date import Date
from pquantlib.time.time_series import TimeSeries


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/indexes/equity")


# Index names this module stores fixings under. Cleared individually rather
# than via clear_histories() so the shared singleton is not wiped for other
# test modules.
_OWNED_HISTORIES = (
    "eqIndexDiv",
    "eqIndexNoDiv",
    "eqIndexNoSpot",
    "eqIndexEmpty",
    "eqIndexNoRate",
    "eqIndexBranches",
    Euribor3M().name(),
)


def _clear_owned_histories() -> None:
    for name in _OWNED_HISTORIES:
        IndexManager().clear_history(name)


@pytest.fixture(autouse=True)
def _clean_state(  # pyright: ignore[reportUnusedFunction]  # pytest auto-uses
    cpp: dict[str, Any],
) -> Iterator[None]:
    """Reset this module's fixings and pin the evaluation date the probe used."""
    _clear_owned_histories()
    previous = ObservableSettings().evaluation_date
    ObservableSettings().evaluation_date = Date(int(cpp["today_serial"]))
    yield
    ObservableSettings().evaluation_date = previous
    _clear_owned_histories()


def _curves(cpp: dict[str, Any]) -> tuple[FlatForward, FlatForward, SimpleQuote]:
    today = Date(int(cpp["today_serial"]))
    interest = FlatForward.from_rate(today, 0.045, Actual365Fixed())
    dividend = FlatForward.from_rate(today, 0.015, Actual365Fixed())
    return interest, dividend, SimpleQuote(cpp["spot"])


def test_wiring_matches_cpp(cpp: dict[str, Any]) -> None:
    interest, dividend, spot = _curves(cpp)
    idx = EquityIndex(
        "eqIndexDiv", UnitedStates(UnitedStates.Market.GovernmentBond),
        USDCurrency(), interest, dividend, spot,
    )
    assert idx.name() == cpp["name_with_dividend"]
    assert idx.fixing_calendar().name() == cpp["fixing_calendar"]
    assert idx.currency().code == cpp["currency_code"]
    assert idx.spot() is spot
    assert idx.equity_interest_rate_curve() is interest
    assert idx.equity_dividend_curve() is dividend


def test_forecast_with_dividend_curve(cpp: dict[str, Any]) -> None:
    interest, dividend, spot = _curves(cpp)
    idx = EquityIndex(
        "eqIndexDiv", UnitedStates(UnitedStates.Market.GovernmentBond),
        USDCurrency(), interest, dividend, spot,
    )
    tight(idx.forecast_fixing(Date(int(cpp["future_serial"]))), cpp["forecast_with_dividend"])


def test_forecast_without_dividend_curve(cpp: dict[str, Any]) -> None:
    """Dropping the dividend leg must change the forward, not merely be tolerated."""
    interest, _, spot = _curves(cpp)
    idx = EquityIndex(
        "eqIndexNoDiv", UnitedStates(UnitedStates.Market.GovernmentBond),
        USDCurrency(), interest, None, spot,
    )
    tight(
        idx.forecast_fixing(Date(int(cpp["future_serial"]))),
        cpp["forecast_without_dividend"],
    )
    assert cpp["forecast_without_dividend"] != cpp["forecast_with_dividend"]


def test_forecast_resolves_spot_from_history_when_quote_absent(cpp: dict[str, Any]) -> None:
    interest, dividend, _ = _curves(cpp)
    cal = UnitedStates(UnitedStates.Market.GovernmentBond)
    idx = EquityIndex("eqIndexNoSpot", cal, USDCurrency(), interest, dividend, None)
    today = Date(int(cpp["today_serial"]))
    idx.add_fixing(cal.adjust(today), cpp["forecast_spotless_last_fixing"])
    tight(idx.forecast_fixing(Date(int(cpp["future_serial"]))), cpp["forecast_spotless"])


def test_forecast_without_spot_or_history_fails(cpp: dict[str, Any]) -> None:
    interest, dividend, _ = _curves(cpp)
    idx = EquityIndex(
        "eqIndexEmpty", UnitedStates(UnitedStates.Market.GovernmentBond),
        USDCurrency(), interest, dividend, None,
    )
    with pytest.raises(LibraryException, match="missing both spot and historical index"):
        idx.forecast_fixing(Date(int(cpp["future_serial"])))


def test_forecast_without_interest_curve_fails(cpp: dict[str, Any]) -> None:
    _, dividend, spot = _curves(cpp)
    idx = EquityIndex(
        "eqIndexNoRate", UnitedStates(UnitedStates.Market.GovernmentBond),
        USDCurrency(), None, dividend, spot,
    )
    with pytest.raises(LibraryException, match="null interest rate term structure"):
        idx.forecast_fixing(Date(int(cpp["future_serial"])))


def test_fixing_branch_selection(cpp: dict[str, Any]) -> None:
    """Future forecasts, past reads history, today falls back to spot."""
    interest, dividend, spot = _curves(cpp)
    idx = EquityIndex(
        "eqIndexBranches", UnitedStates(UnitedStates.Market.GovernmentBond),
        USDCurrency(), interest, dividend, spot,
    )
    past = Date(int(cpp["past_serial"]))
    idx.add_fixing(past, cpp["fixing_past"])

    tight(idx.fixing(Date(int(cpp["future_serial"]))), cpp["fixing_future"])
    tight(idx.fixing(past), cpp["fixing_past"])
    today = Date(int(cpp["today_serial"]))
    tight(idx.fixing(today), cpp["fixing_today_falls_back_to_spot"])
    tight(idx.fixing(today, forecast_todays_fixing=True), cpp["fixing_today_forecast"])


def test_add_fixings_commits_valid_entries_before_raising(cpp: dict[str, Any]) -> None:
    """The valid entries of a partly-invalid batch are stored, then it raises."""
    idx = Euribor3M()
    d0 = Date(int(cpp["batch_d0_serial"]))
    bad = Date(int(cpp["batch_bad_serial"]))
    d1 = Date(int(cpp["batch_d1_serial"]))
    assert idx.is_valid_fixing_date(bad) is cpp["batch_bad_is_valid_fixing_date"]

    assert cpp["batch_threw"] is True
    with pytest.raises(LibraryException, match="invalid fixing"):
        idx.add_fixings([d0, bad, d1], [10.0, 99.0, 12.0])

    series = idx.time_series()
    assert series[d0] == cpp["batch_stored_d0"]
    assert series[d1] == cpp["batch_stored_d1"]
    assert series[bad] is None


def test_readding_the_same_value_is_accepted(cpp: dict[str, Any]) -> None:
    idx = Euribor3M()
    d0 = Date(int(cpp["batch_d0_serial"]))
    idx.add_fixing(d0, 10.0)
    assert cpp["readding_same_value_throws"] is False
    idx.add_fixing(d0, 10.0)
    assert idx.time_series()[d0] == 10.0


def test_readding_a_different_value_raises(cpp: dict[str, Any]) -> None:
    idx = Euribor3M()
    d0 = Date(int(cpp["batch_d0_serial"]))
    idx.add_fixing(d0, 10.0)
    assert cpp["readding_different_value_throws"] is True
    with pytest.raises(LibraryException, match="duplicated fixing"):
        idx.add_fixing(d0, 11.0)


def test_force_overwrite_replaces_the_stored_value(cpp: dict[str, Any]) -> None:
    idx = Euribor3M()
    d0 = Date(int(cpp["batch_d0_serial"]))
    idx.add_fixing(d0, 10.0)
    idx.add_fixing(d0, 11.0, force_overwrite=True)
    assert idx.time_series()[d0] == cpp["force_overwrite_wins"]


def test_add_fixings_from_time_series_round_trips(cpp: dict[str, Any]) -> None:
    idx = Euribor3M()
    d0 = Date(int(cpp["batch_d0_serial"]))
    d1 = Date(int(cpp["batch_d1_serial"]))
    series: TimeSeries[float] = TimeSeries[float].from_pairs([d0, d1], [10.0, 12.0])
    idx.add_fixings_from_time_series(series)
    assert idx.time_series()[d0] == 10.0
    assert idx.time_series()[d1] == 12.0
