"""Tests for ql/prices.{hpp,cpp} — PriceType, mid helpers, IntervalPrice.

Every expected value comes from ``migration-harness/cpp/probes/v143_root_tail/probe.cpp``
run against C++ QuantLib v1.43; the reference lives at
``migration-harness/references/v143/root/tail.json``.

No global evaluation date is involved in this module (the probe's
prices section runs before any ``Settings`` mutation, probe.cpp:539), so no
settings fixture is needed here.
"""

from __future__ import annotations

from typing import Any, Final

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.prices import (
    IntervalPrice,
    IntervalPriceType,
    PriceType,
    mid_equivalent,
    mid_safe,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.time_series import TimeSeries

_TYPE_BY_NAME: Final[dict[str, IntervalPriceType]] = {
    "Open": IntervalPriceType.Open,
    "Close": IntervalPriceType.Close,
    "High": IntervalPriceType.High,
    "Low": IntervalPriceType.Low,
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/root/tail")


def test_price_type_enum_values() -> None:
    """C++ ``enum PriceType`` declaration order (ql/prices.hpp:35-58)."""
    assert [
        PriceType.Bid,
        PriceType.Ask,
        PriceType.Last,
        PriceType.Close,
        PriceType.Mid,
        PriceType.MidEquivalent,
        PriceType.MidSafe,
    ] == [0, 1, 2, 3, 4, 5, 6]


def test_mid_equivalent_full_lattice(cpp: dict[str, Any]) -> None:
    """All 256 availability/positivity combinations, against C++."""
    rows = cpp["mid_equivalent"]
    assert len(rows) == 256
    for row in rows:
        args = (row["bid"], row["ask"], row["last"], row["close"])
        if row["throws"]:
            with pytest.raises(LibraryException):
                mid_equivalent(*args)
        else:
            tolerance.exact(mid_equivalent(*args), row["value"])


def test_mid_safe_lattice(cpp: dict[str, Any]) -> None:
    rows = cpp["mid_safe"]
    assert len(rows) == 16
    for row in rows:
        if row["throws"]:
            with pytest.raises(LibraryException):
                mid_safe(row["bid"], row["ask"])
        else:
            tolerance.exact(mid_safe(row["bid"], row["ask"]), row["value"])


def test_interval_price_default_ctor(cpp: dict[str, Any]) -> None:
    exp = cpp["interval_price"]["default_ctor"]
    p = IntervalPrice()
    assert p.open() == exp["open"]
    assert p.close() == exp["close"]
    assert p.high() == exp["high"]
    assert p.low() == exp["low"]


def test_interval_price_ctor_argument_order(cpp: dict[str, Any]) -> None:
    """C++ ctor order is (open, close, high, low) — not conventional OHLC."""
    exp = cpp["interval_price"]["ctor_open_close_high_low"]
    p = IntervalPrice(11.0, 22.0, 33.0, 44.0)
    tolerance.exact(p.open(), exp["open"])  # pyright: ignore[reportArgumentType]
    tolerance.exact(p.close(), exp["close"])  # pyright: ignore[reportArgumentType]
    tolerance.exact(p.high(), exp["high"])  # pyright: ignore[reportArgumentType]
    tolerance.exact(p.low(), exp["low"])  # pyright: ignore[reportArgumentType]


def test_interval_price_type_enum_values(cpp: dict[str, Any]) -> None:
    exp = cpp["interval_price"]["type_values"]
    for name, value in exp.items():
        assert int(_TYPE_BY_NAME[name]) == value


def test_interval_price_value_by_type(cpp: dict[str, Any]) -> None:
    exp = cpp["interval_price"]["value_by_type"]
    p = IntervalPrice(11.0, 22.0, 33.0, 44.0)
    for name, value in exp.items():
        tolerance.exact(p.value(_TYPE_BY_NAME[name]), value)  # pyright: ignore[reportArgumentType]


def test_interval_price_set_value(cpp: dict[str, Any]) -> None:
    for row in cpp["interval_price"]["set_value"]:
        p = IntervalPrice(11.0, 22.0, 33.0, 44.0)
        p.set_value(99.0, _TYPE_BY_NAME[row["set"]])
        assert [p.open(), p.close(), p.high(), p.low()] == [
            row["open"],
            row["close"],
            row["high"],
            row["low"],
        ]


def test_interval_price_set_values(cpp: dict[str, Any]) -> None:
    exp = cpp["interval_price"]["set_values"]
    p = IntervalPrice(1.0, 2.0, 3.0, 4.0)
    p.set_values(51.0, 52.0, 53.0, 54.0)
    assert [p.open(), p.close(), p.high(), p.low()] == [
        exp["open"],
        exp["close"],
        exp["high"],
        exp["low"],
    ]


def _probe_series(cpp: dict[str, Any]) -> TimeSeries[IntervalPrice]:
    """Rebuild the probe's series (probe.cpp:275-282) — dates NOT sorted on input."""
    dates = [Date(int(s)) for s in cpp["interval_price"]["series_input_dates"]]
    return IntervalPrice.make_series(
        dates,
        [101.0, 201.0, 301.0],
        [102.0, 202.0, 302.0],
        [103.0, 203.0, 303.0],
        [104.0, 204.0, 304.0],
    )


def test_make_series_orders_by_date(cpp: dict[str, Any]) -> None:
    """C++ backs the series with a std::map, so output is key-ordered."""
    ts = _probe_series(cpp)
    assert [d.serial_number() for d in ts.dates()] == cpp["interval_price"]["series_dates"]
    assert cpp["interval_price"]["series_dates"] != cpp["interval_price"]["series_input_dates"]


def test_extract_values(cpp: dict[str, Any]) -> None:
    ts = _probe_series(cpp)
    for name, values in cpp["interval_price"]["extract_values"].items():
        assert IntervalPrice.extract_values(ts, _TYPE_BY_NAME[name]) == values


def test_extract_component(cpp: dict[str, Any]) -> None:
    ts = _probe_series(cpp)
    comp = IntervalPrice.extract_component(ts, IntervalPriceType.Close)
    got = [{"date": d.serial_number(), "value": v} for d, v in comp.items()]
    assert got == cpp["interval_price"]["extract_component_close"]


def test_make_series_size_mismatch_raises(cpp: dict[str, Any]) -> None:
    assert cpp["interval_price"]["make_series_size_mismatch_throws"] is True
    dates = [Date(int(s)) for s in cpp["interval_price"]["series_input_dates"]]
    with pytest.raises(LibraryException):
        IntervalPrice.make_series(dates, [1.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])


def test_interval_price_value_rejects_unknown_type() -> None:
    """C++ ``default: QL_FAIL("Unknown price type")`` (ql/prices.cpp:99).

    ``IntervalPriceType(99)`` is not constructible in Python, so the only way
    to reach the C++ default branch is an out-of-band raw int — which is what
    an unchecked C++ ``Type`` cast amounts to.
    """
    p = IntervalPrice(1.0, 2.0, 3.0, 4.0)
    with pytest.raises(LibraryException):
        p.value(99)  # pyright: ignore[reportArgumentType]
    with pytest.raises(LibraryException):
        p.set_value(1.0, 99)  # pyright: ignore[reportArgumentType]
