"""Tests for LastFixingQuote (cross-validated vs C++ v1.43).

# C++ parity: ql/quotes/lastfixingquote.{hpp,cpp} @ v1.43.

Every expected value comes from
``migration-harness/cpp/probes/v143_quotes_tail/probe.cpp`` (section
``sectionLastFixing``), captured in
``migration-harness/references/v143/quotes/tail.json``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final

import pytest

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.last_fixing_quote import LastFixingQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF: Final[dict[str, Any]] = reference_reader.load("v143/quotes/tail")

# Probe: migration-harness/cpp/probes/v143_quotes_tail/probe.cpp:119
# (`const Date kToday(15, June, 2026);`), set as the evaluation date at
# probe.cpp:628. referenceDate() is min(lastFixingDate, evaluationDate), so
# every assertion here is wall-clock dependent without the pin below.
TODAY: Final[Date] = Date.from_ymd(15, Month.June, 2026)
# probe.cpp:541 — inside the fixing window, so min() picks the evaluation date.
MID_WINDOW: Final[Date] = Date.from_ymd(9, Month.June, 2026)
# probe.cpp:522-527 — every TARGET business day from 1 to 12 June 2026, each
# fixing one basis point above the last, starting at 3.01%.
FIXING_DAYS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 8, 9, 10, 11, 12)
FIRST_FIXING: Final[float] = 0.0301
FIXING_STEP: Final[float] = 0.0001

_INDEX_FAMILY: Final[str] = "LFQtest"
_EMPTY_INDEX_FAMILY: Final[str] = "LFQempty"


@pytest.fixture(autouse=True)
def _pinned_settings() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = TODAY
    try:
        yield
    finally:
        settings.evaluation_date = previous


def _index(family: str) -> IborIndex:
    return IborIndex(
        family,
        Period(6, TimeUnit.Months),
        2,
        EURCurrency(),
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        False,
        Actual360(),
        FlatForward.from_rate(TODAY, 0.0325, Actual365Fixed()),
    )


@pytest.fixture
def index() -> Iterator[IborIndex]:
    """The probe's index and its ten-day fixing history.

    The values are accumulated by repeated ``+=`` exactly as the probe does,
    so the stored doubles are bit-identical to the C++ ones.
    """
    idx = _index(_INDEX_FAMILY)
    idx.clear_fixings()
    value = FIRST_FIXING
    for day in FIXING_DAYS:
        idx.add_fixing(Date.from_ymd(day, Month.June, 2026), value)
        value += FIXING_STEP
    try:
        yield idx
    finally:
        idx.clear_fixings()


@pytest.fixture
def empty_index() -> Iterator[IborIndex]:
    idx = _index(_EMPTY_INDEX_FAMILY)
    idx.clear_fixings()
    try:
        yield idx
    finally:
        idx.clear_fixings()


def test_the_index_name_matches_the_probe(index: IborIndex) -> None:
    """Guards the fixture: a different name would read a different history."""
    assert index.name() == _REF["lfq_index_name"]


def test_the_fixture_history_ends_where_the_probe_history_ends(index: IborIndex) -> None:
    assert index.time_series().last_date().serial_number() == _REF["lfq_last_fixing_date_serial"]


def test_an_evaluation_date_after_the_history_reads_the_last_fixing(index: IborIndex) -> None:
    quote = LastFixingQuote(index)
    assert quote.reference_date().serial_number() == _REF["lfq_after_reference_serial"]
    exact(quote.value(), _REF["lfq_after_value"])
    assert quote.is_valid() is _REF["lfq_after_is_valid"]
    assert quote.index() is index


def test_an_evaluation_date_inside_the_history_reads_back_through_it(index: IborIndex) -> None:
    """``min()`` picks the evaluation date, not the end of the series."""
    ObservableSettings().evaluation_date = MID_WINDOW
    quote = LastFixingQuote(index)
    assert quote.reference_date().serial_number() == _REF["lfq_before_reference_serial"]
    exact(quote.value(), _REF["lfq_before_value"])
    assert quote.is_valid() is _REF["lfq_before_is_valid"]


def test_the_two_evaluation_dates_give_different_fixings() -> None:
    """Guards the two cases above: the ``min()`` really is observable."""
    assert _REF["lfq_after_value"] != _REF["lfq_before_value"]
    assert _REF["lfq_after_reference_serial"] != _REF["lfq_before_reference_serial"]


def test_an_index_with_no_history_is_invalid(empty_index: IborIndex) -> None:
    quote = LastFixingQuote(empty_index)
    assert quote.is_valid() is _REF["lfq_empty_is_valid"]
    assert _REF["lfq_empty_value_raises"] is True
    with pytest.raises(LibraryException, match="has no fixing"):
        quote.value()


def test_the_quote_relays_index_notifications(index: IborIndex) -> None:
    quote = LastFixingQuote(index)
    counts = [0]

    class _Counter:
        def update(self) -> None:
            counts[0] += 1

    observer = _Counter()
    quote.register_with(observer)
    index.update()
    assert counts[0] == 1
