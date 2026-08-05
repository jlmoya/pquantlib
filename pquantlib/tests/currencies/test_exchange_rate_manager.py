"""Cross-validate ExchangeRateManager against C++ QuantLib v1.43.

Probe: ``currencies/exchangerate`` (the ``manager`` and ``money`` sections).

The manager is real logic, not data: ``lookup`` prefers a direct rate, then
routes through the source's or the target's triangulation currency, and only
then walks the whole store for a chain — and each of those branches has a
failing counterpart (wrong date, Direct-only, no route at all) that a port can
get wrong while still looking right on the happy path. Every branch therefore
appears here in both directions.

The ``money`` cases exercise the two Money conversion modes, which go through
``lookup`` and then round in the target currency — EUR being the only currency
with a non-default rounding, that is where a wrong rounding would show.

Tier: EXACT — the arithmetic is a handful of IEEE-754 operations on literal
rates, so a correct port reproduces C++ bit for bit.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.currencies.america import PEHCurrency, PENCurrency, USDCurrency
from pquantlib.currencies.asia import JPYCurrency
from pquantlib.currencies.europe import ATSCurrency, DEMCurrency, EURCurrency, GRDCurrency
from pquantlib.currencies.exchange_rate import ExchangeRate, ExchangeRateType
from pquantlib.currencies.exchange_rate_manager import Entry, ExchangeRateManager
from pquantlib.currencies.money import ConversionType, Money, MoneySettings
from pquantlib.exceptions import LibraryException
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_IN_RANGE = Date.from_ymd(15, Month.June, 2010)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("currencies/exchangerate")["manager"]


@pytest.fixture(scope="module")
def cpp_money() -> dict[str, Any]:
    return reference_reader.load("currencies/exchangerate")["money"]


@pytest.fixture(autouse=True)
def manager() -> Iterator[ExchangeRateManager]:
    """A manager holding exactly the seeded known rates, restored afterwards."""
    mgr = ExchangeRateManager.instance()
    mgr.clear()
    yield mgr
    mgr.clear()


@pytest.fixture
def evaluation_date() -> Iterator[ObservableSettings]:
    settings = ObservableSettings()
    saved = settings.evaluation_date
    yield settings
    settings.evaluation_date = saved


@pytest.fixture
def money_settings() -> Iterator[MoneySettings]:
    settings = MoneySettings.instance()
    saved_type = settings.conversion_type
    saved_base = settings.base_currency
    yield settings
    settings.conversion_type = saved_type
    settings.base_currency = saved_base


def _assert_rate(actual: ExchangeRate, expected: dict[str, Any]) -> None:
    assert actual.source.code == expected["source"]
    assert actual.target.code == expected["target"]
    assert actual.rate is not None
    exact(actual.rate, expected["rate"])
    assert int(actual.type) == expected["type"]


def _assert_money(actual: Money, expected: dict[str, Any]) -> None:
    exact(actual.value, expected["value"])
    assert actual.currency.code == expected["currency"]


# --- lookup branches ----------------------------------------------------


@pytest.mark.exact
def test_same_currency_is_unity(manager: ExchangeRateManager, cpp: dict[str, Any]) -> None:
    _assert_rate(manager.lookup(EURCurrency(), EURCurrency(), _IN_RANGE), cpp["same_currency"])


@pytest.mark.exact
def test_direct_lookup(manager: ExchangeRateManager, cpp: dict[str, Any]) -> None:
    rate = manager.lookup(EURCurrency(), ATSCurrency(), _IN_RANGE, ExchangeRateType.Direct)
    _assert_rate(rate, cpp["direct_eur_ats"])


@pytest.mark.exact
def test_reversed_lookup_returns_the_stored_orientation(
    manager: ExchangeRateManager, cpp: dict[str, Any]
) -> None:
    # Asking ATS -> EUR yields the stored EUR -> ATS rate; it is exchange()
    # that knows which way to apply it, not lookup().
    _assert_rate(manager.lookup(ATSCurrency(), EURCurrency(), _IN_RANGE), cpp["reversed_ats_eur"])


def test_dated_entry_before_its_start_raises(manager: ExchangeRateManager, cpp: dict[str, Any]) -> None:
    assert cpp["dated_grd_before_start"]["raises"] is True
    with pytest.raises(LibraryException):
        manager.lookup(EURCurrency(), GRDCurrency(), Date.from_ymd(1, Month.June, 2000))


@pytest.mark.exact
def test_dated_entry_after_its_start(manager: ExchangeRateManager, cpp: dict[str, Any]) -> None:
    rate = manager.lookup(EURCurrency(), GRDCurrency(), Date.from_ymd(1, Month.June, 2001))
    _assert_rate(rate, cpp["dated_grd_after_start"])


@pytest.mark.exact
def test_triangulated_lookup(manager: ExchangeRateManager, cpp: dict[str, Any]) -> None:
    # ATS and DEM both triangulate through EUR, so the chain is ATS -> EUR -> DEM.
    _assert_rate(manager.lookup(ATSCurrency(), DEMCurrency(), _IN_RANGE), cpp["triangulated_ats_dem"])


def test_direct_refuses_to_triangulate(manager: ExchangeRateManager, cpp: dict[str, Any]) -> None:
    assert cpp["direct_only_ats_dem"]["raises"] is True
    with pytest.raises(LibraryException):
        manager.lookup(ATSCurrency(), DEMCurrency(), _IN_RANGE, ExchangeRateType.Direct)


@pytest.mark.exact
def test_smart_lookup_chains_through_the_store(manager: ExchangeRateManager, cpp: dict[str, Any]) -> None:
    # Neither PEH nor PEN triangulates, so the manager walks its store and
    # finds PEH -> PEI -> PEN.
    _assert_rate(manager.lookup(PEHCurrency(), PENCurrency(), _IN_RANGE), cpp["smart_peh_pen"])


def test_unreachable_pair_raises(manager: ExchangeRateManager, cpp: dict[str, Any]) -> None:
    assert cpp["unreachable_usd_jpy"]["raises"] is True
    with pytest.raises(LibraryException):
        manager.lookup(USDCurrency(), JPYCurrency(), _IN_RANGE)


# --- default date -------------------------------------------------------


@pytest.mark.exact
def test_default_date_uses_the_evaluation_date(
    manager: ExchangeRateManager,
    evaluation_date: ObservableSettings,
    cpp: dict[str, Any],
) -> None:
    evaluation_date.evaluation_date = Date.from_ymd(1, Month.June, 2001)
    _assert_rate(manager.lookup(EURCurrency(), GRDCurrency()), cpp["default_date_after_start"])


def test_default_date_respects_the_entry_range(
    manager: ExchangeRateManager,
    evaluation_date: ObservableSettings,
    cpp: dict[str, Any],
) -> None:
    assert cpp["default_date_before_start"]["raises"] is True
    evaluation_date.evaluation_date = Date.from_ymd(1, Month.June, 2000)
    with pytest.raises(LibraryException):
        manager.lookup(EURCurrency(), GRDCurrency())


# --- user-added rates ---------------------------------------------------


def _add_user_rate(manager: ExchangeRateManager, rate: float) -> None:
    manager.add(
        ExchangeRate(USDCurrency(), EURCurrency(), rate),
        Date.from_ymd(1, Month.January, 2020),
        Date.from_ymd(31, Month.December, 2020),
    )


@pytest.mark.exact
def test_user_added_rate_inside_its_range(manager: ExchangeRateManager, cpp: dict[str, Any]) -> None:
    _add_user_rate(manager, 1.25)
    rate = manager.lookup(
        USDCurrency(), EURCurrency(), Date.from_ymd(15, Month.June, 2020), ExchangeRateType.Direct
    )
    _assert_rate(rate, cpp["user_added_in_range"])


@pytest.mark.parametrize(
    ("case", "year"),
    [("user_added_before_range", 2019), ("user_added_after_range", 2021)],
)
def test_user_added_rate_outside_its_range_raises(
    manager: ExchangeRateManager, cpp: dict[str, Any], case: str, year: int
) -> None:
    assert cpp[case]["raises"] is True
    _add_user_rate(manager, 1.25)
    with pytest.raises(LibraryException):
        manager.lookup(
            USDCurrency(),
            EURCurrency(),
            Date.from_ymd(15, Month.June, year),
            ExchangeRateType.Direct,
        )


@pytest.mark.exact
def test_latest_overlapping_rate_wins(manager: ExchangeRateManager, cpp: dict[str, Any]) -> None:
    _add_user_rate(manager, 1.25)
    _add_user_rate(manager, 1.5)
    rate = manager.lookup(
        USDCurrency(), EURCurrency(), Date.from_ymd(15, Month.June, 2020), ExchangeRateType.Direct
    )
    _assert_rate(rate, cpp["user_added_latest_wins"])


def test_clear_drops_user_rates(manager: ExchangeRateManager, cpp: dict[str, Any]) -> None:
    assert cpp["cleared_drops_user_rate"]["raises"] is True
    _add_user_rate(manager, 1.25)
    manager.clear()
    with pytest.raises(LibraryException):
        manager.lookup(
            USDCurrency(),
            EURCurrency(),
            Date.from_ymd(15, Month.June, 2020),
            ExchangeRateType.Direct,
        )


@pytest.mark.exact
def test_clear_restores_the_known_rates(manager: ExchangeRateManager, cpp: dict[str, Any]) -> None:
    _add_user_rate(manager, 1.25)
    manager.clear()
    rate = manager.lookup(EURCurrency(), ATSCurrency(), _IN_RANGE, ExchangeRateType.Direct)
    _assert_rate(rate, cpp["cleared_restores_known"])


# --- Money conversion ---------------------------------------------------


@pytest.mark.exact
def test_automated_conversion_sum(
    manager: ExchangeRateManager,
    money_settings: MoneySettings,
    cpp_money: dict[str, Any],
) -> None:
    money_settings.conversion_type = ConversionType.AUTOMATED_CONVERSION
    total = Money(100.0, ATSCurrency()) + Money(1.0, EURCurrency())
    _assert_money(total, cpp_money["automated_conversion_sum"])


@pytest.mark.exact
def test_base_currency_conversion_sum(
    manager: ExchangeRateManager,
    money_settings: MoneySettings,
    cpp_money: dict[str, Any],
) -> None:
    # Converting into EUR rounds to 2 decimals on the way, which is the whole
    # point of the case: EUR is the only currency with a non-default rounding.
    money_settings.conversion_type = ConversionType.BASE_CURRENCY_CONVERSION
    money_settings.base_currency = EURCurrency()
    total = Money(100.0, ATSCurrency()) + Money(1.0, EURCurrency())
    _assert_money(total, cpp_money["base_currency_conversion_sum"])


def test_base_currency_conversion_comparison(
    manager: ExchangeRateManager,
    money_settings: MoneySettings,
    cpp_money: dict[str, Any],
) -> None:
    money_settings.conversion_type = ConversionType.BASE_CURRENCY_CONVERSION
    money_settings.base_currency = EURCurrency()
    less = Money(100.0, ATSCurrency()) < Money(100.0, EURCurrency())
    assert less is cpp_money["base_currency_conversion_less"]


def test_no_conversion_still_refuses_mixed_currencies(
    manager: ExchangeRateManager,
    money_settings: MoneySettings,
    cpp_money: dict[str, Any],
) -> None:
    assert cpp_money["no_conversion_sum"]["raises"] is True
    money_settings.conversion_type = ConversionType.NO_CONVERSION
    with pytest.raises(LibraryException):
        _ = Money(100.0, ATSCurrency()) + Money(1.0, EURCurrency())


def test_base_currency_conversion_needs_a_base_currency(
    manager: ExchangeRateManager, money_settings: MoneySettings
) -> None:
    # C++ parity: convertToBase QL_REQUIREs a non-empty base currency.
    money_settings.conversion_type = ConversionType.BASE_CURRENCY_CONVERSION
    with pytest.raises(LibraryException, match="no base currency set"):
        _ = Money(100.0, ATSCurrency()) + Money(1.0, EURCurrency())


# --- API semantics not worth a probe case -------------------------------


def test_entry_is_inclusive_on_both_ends() -> None:
    start = Date.from_ymd(1, Month.January, 2020)
    end = Date.from_ymd(31, Month.December, 2020)
    entry = Entry(ExchangeRate(USDCurrency(), EURCurrency(), 1.25), start, end)
    assert entry.valid_at(start)
    assert entry.valid_at(end)
    assert not entry.valid_at(start - 1)
    assert not entry.valid_at(end + 1)


def test_entry_is_reachable_as_a_nested_name() -> None:
    # C++ parity: ExchangeRateManager::Entry.
    assert ExchangeRateManager.Entry is Entry


def test_manager_is_a_singleton() -> None:
    assert ExchangeRateManager.instance() is ExchangeRateManager()
