"""Tests for ForwardValueQuote (cross-validated vs C++ v1.43).

# C++ parity: ql/quotes/forwardvaluequote.{hpp,cpp} @ v1.43.

Every expected value comes from
``migration-harness/cpp/probes/v143_quotes_tail/probe.cpp`` (section
``sectionForwardValue``), captured in
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
from pquantlib.quotes.forward_value_quote import ForwardValueQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF: Final[dict[str, Any]] = reference_reader.load("v143/quotes/tail")

# Probe: migration-harness/cpp/probes/v143_quotes_tail/probe.cpp:119
# (`const Date kToday(15, June, 2026);`), set as the evaluation date at
# probe.cpp:628. Index::fixing splits on the evaluation date, so this module
# is wall-clock dependent without the pin below.
TODAY: Final[Date] = Date.from_ymd(15, Month.June, 2026)
# probe.cpp:376 — a stored historical fixing, before TODAY.
PAST_FIXING_DATE: Final[Date] = Date.from_ymd(5, Month.June, 2026)
PAST_FIXING: Final[float] = 0.0287
# probe.cpp:380 — after TODAY, so the index forecasts off the curve.
FORECAST_DATE: Final[Date] = Date.from_ymd(17, Month.September, 2026)
# probe.cpp:397 — a Sunday: not a valid fixing date for TARGET.
UNFIXABLE_DATE: Final[Date] = Date.from_ymd(14, Month.June, 2026)

_INDEX_FAMILY: Final[str] = "FVQtest"


@pytest.fixture(autouse=True)
def _pinned_settings() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = TODAY
    try:
        yield
    finally:
        settings.evaluation_date = previous


@pytest.fixture
def index() -> Iterator[IborIndex]:
    """The probe's index (probe.cpp:345-352, 373-377), with its history.

    Fixings live in the process-global IndexManager keyed on the index name, so
    the history is cleared on the way out.
    """
    idx = IborIndex(
        _INDEX_FAMILY,
        Period(6, TimeUnit.Months),
        2,
        EURCurrency(),
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        False,
        Actual360(),
        FlatForward.from_rate(TODAY, 0.0325, Actual365Fixed()),
    )
    idx.clear_fixings()
    idx.add_fixing(PAST_FIXING_DATE, PAST_FIXING)
    try:
        yield idx
    finally:
        idx.clear_fixings()


def test_the_index_name_matches_the_probe(index: IborIndex) -> None:
    """Guards the fixture: a different name would read a different history."""
    assert index.name() == _REF["fvq_index_name"]


def test_a_future_fixing_date_is_forecast_off_the_curve(index: IborIndex) -> None:
    quote = ForwardValueQuote(index, FORECAST_DATE)
    assert FORECAST_DATE.serial_number() == _REF["fvq_forecast_fixing_serial"]
    tight(quote.value(), _REF["fvq_forecast_value"])
    assert quote.is_valid() is _REF["fvq_forecast_is_valid"]


def test_a_past_fixing_date_is_served_from_the_history(index: IborIndex) -> None:
    quote = ForwardValueQuote(index, PAST_FIXING_DATE)
    assert PAST_FIXING_DATE.serial_number() == _REF["fvq_past_fixing_serial"]
    exact(quote.value(), _REF["fvq_past_value"])
    assert quote.is_valid() is _REF["fvq_past_is_valid"]


def test_the_two_fixing_dates_give_different_values() -> None:
    """Guards the two cases above: ``fixing_date`` is observable."""
    assert _REF["fvq_forecast_value"] != _REF["fvq_past_value"]


def test_is_valid_is_true_even_when_value_cannot_be_produced(index: IborIndex) -> None:
    """C++ ``isValid()`` returns a hard ``true`` — the port keeps that.

    The C++ source comments the choice with "not sure this is the best
    approach..."; a quote on a non-business day still reports valid and only
    fails when asked for its value.
    """
    quote = ForwardValueQuote(index, UNFIXABLE_DATE)
    assert quote.is_valid() is _REF["fvq_unfixable_is_valid"]
    assert _REF["fvq_unfixable_value_raises"] is True
    with pytest.raises(LibraryException, match="Fixing date 2026-06-14 is not valid"):
        quote.value()


def test_the_quote_relays_index_notifications(index: IborIndex) -> None:
    quote = ForwardValueQuote(index, FORECAST_DATE)
    counts = [0]

    class _Counter:
        def update(self) -> None:
            counts[0] += 1

    observer = _Counter()
    quote.register_with(observer)
    index.update()
    assert counts[0] == 1
