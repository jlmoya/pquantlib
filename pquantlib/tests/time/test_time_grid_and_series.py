"""Cross-validate TimeGrid + TimeSeries against the C++ probe.

Probe source: migration-harness/cpp/probes/time/timegrid_probe.cpp
Reference:    migration-harness/references/time/timegrid.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid
from pquantlib.time.time_series import TimeSeries


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("time/timegrid")


# --- TimeGrid: regular -----------------------------------------------------


def test_regular_matches_cpp(cpp: dict[str, Any]) -> None:
    tg = TimeGrid.regular(1.0, 4)
    r = cpp["regular"]
    assert tg.size() == int(r["size"])
    for got, expected in zip(tg.times, r["times"], strict=True):
        tolerance.tight(got, float(expected))
    for i, expected in enumerate(r["dt"]):
        tolerance.tight(tg.dt(i), float(expected))
    tolerance.exact(tg.front(), float(r["front"]))
    tolerance.exact(tg.back(), float(r["back"]))


# --- TimeGrid: mandatory-only ---------------------------------------------


def test_mandatory_only_matches_cpp(cpp: dict[str, Any]) -> None:
    tg = TimeGrid.with_mandatory([0.5, 1.0, 1.5, 2.0])
    m = cpp["mandatory_only"]
    assert tg.size() == int(m["size"])
    for got, expected in zip(tg.times, m["times"], strict=True):
        tolerance.tight(got, float(expected))


# --- TimeGrid: mandatory + steps ------------------------------------------


def test_mandatory_with_steps_matches_cpp(cpp: dict[str, Any]) -> None:
    tg = TimeGrid.with_mandatory_and_steps([1.0, 2.0], 4)
    m = cpp["mandatory_with_steps"]
    assert tg.size() == int(m["size"])
    for got, expected in zip(tg.times, m["times"], strict=True):
        tolerance.tight(got, float(expected))


# --- TimeGrid: lookups ----------------------------------------------------


def test_lookups_match_cpp(cpp: dict[str, Any]) -> None:
    tg = TimeGrid.regular(1.0, 4)
    lk = cpp["lookups"]
    assert tg.index(0.5) == int(lk["index_at_0_5"])
    assert tg.closest_index(0.4) == int(lk["closest_index_at_0_4"])
    assert tg.closest_index(0.6) == int(lk["closest_index_at_0_6"])
    assert tg.closest_index(-1.0) == int(lk["closest_index_at_neg"])
    assert tg.closest_index(100.0) == int(lk["closest_index_at_big"])
    tolerance.tight(tg.closest_time(0.4), float(lk["closest_time_at_0_4"]))


def test_index_off_grid_raises() -> None:
    tg = TimeGrid.regular(1.0, 4)
    with pytest.raises(LibraryException, match="inadequate"):
        tg.index(2.0)
    with pytest.raises(LibraryException, match="inadequate"):
        tg.index(0.3)


def test_negative_end_raises() -> None:
    with pytest.raises(LibraryException, match="negative times"):
        TimeGrid.regular(-1.0, 4)


def test_with_mandatory_empty_raises() -> None:
    with pytest.raises(LibraryException, match="empty time sequence"):
        TimeGrid.with_mandatory([])


# --- TimeSeries -----------------------------------------------------------


def test_timeseries_basic_matches_cpp(cpp: dict[str, Any]) -> None:
    ts: TimeSeries[float] = TimeSeries()
    ts[Date.from_ymd(15, Month.March, 2024)] = 1.0
    ts[Date.from_ymd(15, Month.April, 2024)] = 2.0
    ts[Date.from_ymd(15, Month.May, 2024)] = 3.0

    t = cpp["timeseries"]
    assert ts.size() == int(t["size"])
    assert ts.first_date().serial == int(t["first_date_serial"])
    assert ts.last_date().serial == int(t["last_date_serial"])
    tolerance.exact(ts[Date.from_ymd(15, Month.April, 2024)] or 0.0, float(t["value_at_april"]))


def test_timeseries_missing_key_returns_none() -> None:
    ts: TimeSeries[float] = TimeSeries()
    ts[Date.from_ymd(15, Month.March, 2024)] = 1.0
    assert ts[Date.from_ymd(1, Month.January, 2099)] is None


def test_timeseries_from_pairs() -> None:
    dates = [Date.from_ymd(1, Month.January, 2024), Date.from_ymd(2, Month.January, 2024)]
    values = [10.0, 20.0]
    ts = TimeSeries[float].from_pairs(dates, values)
    assert ts.size() == 2
    assert ts[dates[1]] == 20.0


def test_timeseries_from_first_date() -> None:
    start = Date.from_ymd(1, Month.January, 2024)
    ts = TimeSeries[float].from_first_date(start, [1.0, 2.0, 3.0])
    assert ts.size() == 3
    assert ts[start + 2] == 3.0


def test_timeseries_dates_values_items_ordered() -> None:
    ts: TimeSeries[float] = TimeSeries()
    later = Date.from_ymd(1, Month.March, 2024)
    earlier = Date.from_ymd(1, Month.January, 2024)
    ts[later] = 30.0
    ts[earlier] = 10.0
    # Out-of-order inserts, but dates() / values() / items() return sorted.
    assert ts.dates() == (earlier, later)
    assert ts.values() == (10.0, 30.0)
    assert ts.items() == ((earlier, 10.0), (later, 30.0))


def test_timeseries_first_last_on_empty_raises() -> None:
    ts: TimeSeries[float] = TimeSeries()
    with pytest.raises(LibraryException, match="empty time series"):
        ts.first_date()
    with pytest.raises(LibraryException, match="empty time series"):
        ts.last_date()
