"""Tests for ForwardSwapQuote (cross-validated vs C++ v1.43).

# C++ parity: ql/quotes/forwardswapquote.{hpp,cpp} @ v1.43.

Every expected value comes from
``migration-harness/cpp/probes/v143_quotes_tail/probe.cpp`` (section
``sectionForwardSwap``), captured in
``migration-harness/references/v143/quotes/tail.json``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final

import pytest

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.forward_swap_quote import ForwardSwapQuote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF: Final[dict[str, Any]] = reference_reader.load("v143/quotes/tail")

# Probe: migration-harness/cpp/probes/v143_quotes_tail/probe.cpp:119
# (`const Date kToday(15, June, 2026);`), set as the evaluation date at
# probe.cpp:628. initializeDates() snaps all three dates off the evaluation
# date, so every serial below is wall-clock dependent without the pin.
TODAY: Final[Date] = Date.from_ymd(15, Month.June, 2026)
# probe.cpp:122 / :436 — the roll target for the re-snap case.
LATER: Final[Date] = Date.from_ymd(24, Month.June, 2026)

FORWARD_START: Final[Period] = Period(3, TimeUnit.Months)
SPREAD: Final[float] = 0.0025


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
def swap_index() -> SwapIndex:
    """The probe's 5Y-vs-6M EUR swap index (probe.cpp:409-414)."""
    curve = FlatForward.from_rate(TODAY, 0.0325, Actual365Fixed())
    ibor = IborIndex(
        "FSQtest",
        Period(6, TimeUnit.Months),
        2,
        EURCurrency(),
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        False,
        Actual360(),
        curve,
    )
    return SwapIndex(
        "FSQtestSwap",
        Period(5, TimeUnit.Years),
        2,
        EURCurrency(),
        TARGET(),
        Period(1, TimeUnit.Years),
        BusinessDayConvention.Unadjusted,
        Thirty360(Convention.BondBasis),
        ibor,
    )


def _assert_matches(quote: ForwardSwapQuote, case: str) -> None:
    assert quote.value_date().serial_number() == _REF[case + "_value_date_serial"]
    assert quote.start_date().serial_number() == _REF[case + "_start_date_serial"]
    assert quote.fixing_date().serial_number() == _REF[case + "_fixing_date_serial"]
    tight(quote.value(), _REF[case + "_value"])
    assert quote.is_valid() is _REF[case + "_is_valid"]


# --- the spread argument is threaded, not discarded --------------------------


def test_a_non_zero_spread_shifts_the_par_rate(swap_index: SwapIndex) -> None:
    _assert_matches(ForwardSwapQuote(swap_index, SimpleQuote(SPREAD), FORWARD_START), "fsq_spread25bp")


def test_a_zero_spread_leaves_the_par_rate_alone(swap_index: SwapIndex) -> None:
    _assert_matches(ForwardSwapQuote(swap_index, SimpleQuote(0.0), FORWARD_START), "fsq_spread0")


def test_an_empty_spread_handle_behaves_as_a_zero_spread(swap_index: SwapIndex) -> None:
    """C++ ``spread_.empty() ? 0.0 : spread_->value()``, ported as ``None``."""
    _assert_matches(ForwardSwapQuote(swap_index, None, FORWARD_START), "fsq_spread_empty")
    assert _REF["fsq_spread_empty_value"] == _REF["fsq_spread0_value"]


def test_the_spread_is_observable_in_the_value() -> None:
    """Guards the three cases above: a dropped spread would collapse them."""
    assert _REF["fsq_spread25bp_value"] != _REF["fsq_spread0_value"]


# --- the forward-start argument is threaded, not discarded -------------------


def test_a_zero_forward_start_moves_every_date_and_the_value(swap_index: SwapIndex) -> None:
    quote = ForwardSwapQuote(swap_index, SimpleQuote(SPREAD), Period(0, TimeUnit.Days))
    assert quote.value_date().serial_number() == _REF["fsq_fwdstart0_value_date_serial"]
    assert quote.start_date().serial_number() == _REF["fsq_fwdstart0_start_date_serial"]
    assert quote.fixing_date().serial_number() == _REF["fsq_fwdstart0_fixing_date_serial"]
    tight(quote.value(), _REF["fsq_fwdstart0_value"])
    # With no forward start the swap starts on the value date itself.
    assert _REF["fsq_fwdstart0_start_date_serial"] == _REF["fsq_fwdstart0_value_date_serial"]
    assert _REF["fsq_fwdstart0_start_date_serial"] != _REF["fsq_spread25bp_start_date_serial"]


# --- update() re-snaps the dates ---------------------------------------------


def test_rolling_the_evaluation_date_re_snaps_the_dates_and_the_swap(
    swap_index: SwapIndex,
) -> None:
    """The only path that exercises ``update()``'s ``initializeDates()`` call."""
    settings = ObservableSettings()
    quote = ForwardSwapQuote(swap_index, SimpleQuote(SPREAD), FORWARD_START)
    _assert_matches(quote, "fsq_before_roll")

    settings.evaluation_date = LATER
    _assert_matches(quote, "fsq_after_roll")
    assert _REF["fsq_after_roll_value_date_serial"] != _REF["fsq_before_roll_value_date_serial"]
    assert _REF["fsq_after_roll_value"] != _REF["fsq_before_roll_value"]

    settings.evaluation_date = TODAY
    _assert_matches(quote, "fsq_after_roll_back")


# --- validity ----------------------------------------------------------------


def test_an_invalid_spread_makes_the_quote_invalid(swap_index: SwapIndex) -> None:
    quote = ForwardSwapQuote(swap_index, SimpleQuote(), FORWARD_START)
    assert quote.is_valid() is _REF["fsq_invalid_spread_is_valid"]


def test_a_moving_spread_invalidates_the_cached_rate(swap_index: SwapIndex) -> None:
    spread = SimpleQuote(0.0)
    quote = ForwardSwapQuote(swap_index, spread, FORWARD_START)
    tight(quote.value(), _REF["fsq_spread0_value"])
    spread.set_value(SPREAD)
    tight(quote.value(), _REF["fsq_spread25bp_value"])


def test_the_quote_notifies_its_own_observers_when_the_spread_moves(
    swap_index: SwapIndex,
) -> None:
    spread = SimpleQuote(0.0)
    quote = ForwardSwapQuote(swap_index, spread, FORWARD_START)
    counts = [0]

    class _Counter:
        def update(self) -> None:
            counts[0] += 1

    observer = _Counter()
    quote.register_with(observer)
    spread.set_value(SPREAD)
    assert counts[0] == 1
