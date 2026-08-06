"""Tests for FuturesConvAdjustmentQuote (cross-validated vs C++ v1.43).

# C++ parity: ql/quotes/futuresconvadjustmentquote.{hpp,cpp} @ v1.43.

Every expected value comes from
``migration-harness/cpp/probes/v143_quotes_tail/probe.cpp`` (section
``sectionFuturesConvAdj``), captured in
``migration-harness/references/v143/quotes/tail.json``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final

import pytest

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.futures_conv_adjustment_quote import FuturesConvAdjustmentQuote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time import imm
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF: Final[dict[str, Any]] = reference_reader.load("v143/quotes/tail")

# Probe: migration-harness/cpp/probes/v143_quotes_tail/probe.cpp:119
# (`const Date kToday(15, June, 2026);`), set as the evaluation date at
# probe.cpp:628. value() measures both year fractions FROM the evaluation
# date, so the whole module is wall-clock dependent without the pin below.
TODAY: Final[Date] = Date.from_ymd(15, Month.June, 2026)
# probe.cpp:122 / :487 — the roll target for the cache-invalidation case.
LATER: Final[Date] = Date.from_ymd(24, Month.June, 2026)

IMM_CODE: Final[str] = "U6"
FUTURES_PRICE: Final[float] = 97.85
VOLATILITY: Final[float] = 0.011
MEAN_REVERSION: Final[float] = 0.03
ALT_VOLATILITY: Final[float] = 0.02
ALT_MEAN_REVERSION: Final[float] = 0.05

_INDEX_FAMILY: Final[str] = "FCAtest"


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
def index() -> IborIndex:
    """The probe's 3M index (probe.cpp:345-352, :466)."""
    return IborIndex(
        _INDEX_FAMILY,
        Period(3, TimeUnit.Months),
        2,
        EURCurrency(),
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        False,
        Actual360(),
        FlatForward.from_rate(TODAY, 0.0325, Actual365Fixed()),
    )


@pytest.fixture
def imm_date() -> Date:
    return imm.date(IMM_CODE, TODAY)


def _quote(
    index: IborIndex,
    imm_date: Date,
    volatility: float = VOLATILITY,
    mean_reversion: float = MEAN_REVERSION,
    futures_price: float | None = FUTURES_PRICE,
) -> FuturesConvAdjustmentQuote:
    return FuturesConvAdjustmentQuote(
        index,
        imm_date,
        None if futures_price is None else SimpleQuote(futures_price),
        SimpleQuote(volatility),
        SimpleQuote(mean_reversion),
    )


def test_the_probe_setup_is_reproduced(index: IborIndex, imm_date: Date) -> None:
    """Guards every other case: same IMM date, same index maturity."""
    assert imm_date.serial_number() == _REF["fca_imm_code_resolves_to_serial"]
    assert index.maturity_date(imm_date).serial_number() == _REF["fca_index_maturity_serial"]


def test_value_and_inspectors(index: IborIndex, imm_date: Date) -> None:
    quote = _quote(index, imm_date)
    tight(quote.value(), _REF["fca_by_date_value"])
    exact(quote.futures_value(), _REF["fca_by_date_futures_value"])
    exact(quote.volatility(), _REF["fca_by_date_volatility"])
    exact(quote.mean_reversion(), _REF["fca_by_date_mean_reversion"])
    assert quote.imm_date().serial_number() == _REF["fca_by_date_imm_date_serial"]
    assert quote.is_valid() is _REF["fca_by_date_is_valid"]


def test_the_imm_code_factory_agrees_with_the_explicit_date(index: IborIndex) -> None:
    """C++'s second constructor, which delegates via ``IMM::date(immCode)``.

    The C++ default reference date for ``IMM::date`` is the evaluation date;
    ``pquantlib.time.imm.date`` defaults to today's date instead, so
    ``from_imm_code`` passes the evaluation date explicitly. This test is what
    would catch that going wrong — with the evaluation date pinned to 2026, a
    wall-clock reference date resolves "U6" to a different decade.
    """
    quote = FuturesConvAdjustmentQuote.from_imm_code(
        index,
        IMM_CODE,
        SimpleQuote(FUTURES_PRICE),
        SimpleQuote(VOLATILITY),
        SimpleQuote(MEAN_REVERSION),
    )
    tight(quote.value(), _REF["fca_by_imm_code_value"])
    assert quote.imm_date().serial_number() == _REF["fca_by_imm_code_imm_date_serial"]
    assert _REF["fca_by_imm_code_value"] == _REF["fca_by_date_value"]


def test_volatility_and_mean_reversion_are_not_interchangeable(
    index: IborIndex, imm_date: Date
) -> None:
    """Same two numbers, swapped between the two arguments, must not agree."""
    straight = _quote(index, imm_date, ALT_VOLATILITY, ALT_MEAN_REVERSION)
    tight(straight.value(), _REF["fca_alt_params_value"])
    exact(straight.volatility(), _REF["fca_alt_params_volatility"])
    exact(straight.mean_reversion(), _REF["fca_alt_params_mean_reversion"])

    swapped = _quote(index, imm_date, ALT_MEAN_REVERSION, ALT_VOLATILITY)
    tight(swapped.value(), _REF["fca_params_swapped_value"])
    exact(swapped.volatility(), _REF["fca_params_swapped_volatility"])
    exact(swapped.mean_reversion(), _REF["fca_params_swapped_mean_reversion"])

    assert _REF["fca_alt_params_value"] != _REF["fca_params_swapped_value"]
    assert _REF["fca_alt_params_value"] != _REF["fca_by_date_value"]


def test_rolling_the_evaluation_date_invalidates_the_cached_rate(
    index: IborIndex, imm_date: Date
) -> None:
    """``rate_`` is only dropped by the notification from the evaluation date."""
    settings = ObservableSettings()
    quote = _quote(index, imm_date)
    tight(quote.value(), _REF["fca_before_roll_value"])
    settings.evaluation_date = LATER
    tight(quote.value(), _REF["fca_after_roll_value"])
    settings.evaluation_date = TODAY
    tight(quote.value(), _REF["fca_after_roll_back_value"])
    assert _REF["fca_before_roll_value"] != _REF["fca_after_roll_value"]


@pytest.mark.parametrize(
    ("case", "futures", "volatility", "mean_reversion"),
    [
        ("fca_invalid_futures_is_valid", None, SimpleQuote(VOLATILITY), SimpleQuote(MEAN_REVERSION)),
        ("fca_invalid_vol_is_valid", SimpleQuote(FUTURES_PRICE), None, SimpleQuote(MEAN_REVERSION)),
        ("fca_invalid_mr_is_valid", SimpleQuote(FUTURES_PRICE), SimpleQuote(VOLATILITY), None),
    ],
)
def test_any_invalid_input_makes_the_quote_invalid(
    index: IborIndex,
    imm_date: Date,
    case: str,
    futures: SimpleQuote | None,
    volatility: SimpleQuote | None,
    mean_reversion: SimpleQuote | None,
) -> None:
    """``None`` here stands for a *value-less* SimpleQuote, not an empty handle."""
    quote = FuturesConvAdjustmentQuote(
        index,
        imm_date,
        futures if futures is not None else SimpleQuote(),
        volatility if volatility is not None else SimpleQuote(),
        mean_reversion if mean_reversion is not None else SimpleQuote(),
    )
    assert quote.is_valid() is _REF[case]


def test_an_empty_futures_handle_makes_the_quote_invalid(
    index: IborIndex, imm_date: Date
) -> None:
    quote = _quote(index, imm_date, futures_price=None)
    assert quote.is_valid() is _REF["fca_empty_futures_is_valid"]


def test_an_input_quote_moving_invalidates_the_cache(index: IborIndex, imm_date: Date) -> None:
    volatility = SimpleQuote(VOLATILITY)
    quote = FuturesConvAdjustmentQuote(
        index, imm_date, SimpleQuote(FUTURES_PRICE), volatility, SimpleQuote(MEAN_REVERSION)
    )
    tight(quote.value(), _REF["fca_by_date_value"])
    volatility.set_value(ALT_VOLATILITY)
    # Same numbers as fca_alt_params except for the mean reversion, so this is
    # a distinct value that only a genuinely invalidated cache can produce.
    assert quote.value() != _REF["fca_by_date_value"]
