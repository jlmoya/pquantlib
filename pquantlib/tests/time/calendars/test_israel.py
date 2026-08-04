"""Cross-validate the four Israel markets' holidays for 2020-2030 against the C++ probe.

Probe keys: time/calendars/all -> "israel" (default = TASE), "israel_tase",
"israel_shir", "israel_telbor".

Only the default market was covered before, so a divergence in SHIR — or the
absence of Telbor, added in v1.43 — was structurally invisible.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.testing import reference_reader
from pquantlib.time.calendars.israel import Israel, IsraelMarket
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("time/calendars/all")


_MARKETS: list[tuple[str, IsraelMarket | None]] = [
    ("israel", None),  # default-constructed
    ("israel_tase", IsraelMarket.TASE),
    ("israel_shir", IsraelMarket.SHIR),
    ("israel_telbor", IsraelMarket.Telbor),
]


def _calendar(market: IsraelMarket | None) -> Israel:
    return Israel() if market is None else Israel(market)


@pytest.mark.parametrize(("key", "market"), _MARKETS)
def test_name_matches_cpp(cpp: dict[str, Any], key: str, market: IsraelMarket | None) -> None:
    assert _calendar(market).name() == cpp[key]["name"]


@pytest.mark.parametrize(("key", "market"), _MARKETS)
def test_holidays_match_cpp(cpp: dict[str, Any], key: str, market: IsraelMarket | None) -> None:
    cal = _calendar(market)
    expected = {
        Date.from_ymd(int(h["d"]), Month(int(h["m"])), int(h["y"])) for h in cpp[key]["holidays"]
    }
    actual: set[Date] = set()
    for year in range(2020, 2031):
        for d in cal.holiday_list(
            Date.from_ymd(1, Month.January, year),
            Date.from_ymd(31, Month.December, year),
            include_weekends=False,
        ):
            actual.add(d)
    assert actual == expected, f"diff: missing={expected - actual}, extra={actual - expected}"


def test_telbor_weekend_is_saturday_sunday() -> None:
    """C++ ``Israel::TelborImpl::isWeekend`` (israel.cpp:473-475)."""
    cal = Israel(IsraelMarket.Telbor)
    for w in Weekday:
        assert cal.is_weekend(w) == (w in (Weekday.Saturday, Weekday.Sunday))


def test_telbor_observes_no_good_friday() -> None:
    """Telbor derives from ``Calendar::Impl``, SHIR from ``WesternImpl``.

    The two fixing calendars are otherwise near-identical, so the Easter-derived
    holidays are the sharpest way to tell them apart. Good Friday 2023 was
    7-April; SHIR closes, Telbor does not.
    """
    good_friday_2023 = Date.from_ymd(7, Month.April, 2023)
    assert Israel(IsraelMarket.SHIR).is_holiday(good_friday_2023)
    assert Israel(IsraelMarket.Telbor).is_business_day(good_friday_2023)


def test_settlement_market_is_deprecated() -> None:
    """C++ v1.43 marked ``Israel::Settlement`` ``[[deprecated]]``."""
    with pytest.deprecated_call():
        Israel(IsraelMarket.Settlement)
