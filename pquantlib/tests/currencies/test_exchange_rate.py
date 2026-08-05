"""Cross-validate ExchangeRate against C++ QuantLib v1.43.

Probe: ``currencies/exchangerate`` (the ``exchange_rate`` section).

Covers applying a Direct rate in both directions, the four orientations of
``chain`` (which decide the resulting source/target pair and whether the rate
is a ratio, a product or a reciprocal), and exchanging through a Derived rate.

Tier: EXACT — every value here is one or two IEEE-754 operations on literals,
so a correct port reproduces C++ bit for bit; anything looser would hide a
wrong chain orientation that happens to be numerically close.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.currencies.asia import JPYCurrency
from pquantlib.currencies.currency import Currency
from pquantlib.currencies.europe import EURCurrency, GBPCurrency
from pquantlib.currencies.exchange_rate import ExchangeRate, ExchangeRateType
from pquantlib.currencies.money import Money
from pquantlib.exceptions import LibraryException
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("currencies/exchangerate")["exchange_rate"]


def _assert_rate(actual: ExchangeRate, expected: dict[str, Any]) -> None:
    assert actual.source.code == expected["source"]
    assert actual.target.code == expected["target"]
    assert actual.rate is not None
    exact(actual.rate, expected["rate"])
    assert int(actual.type) == expected["type"]


def _assert_money(actual: Money, expected: dict[str, Any]) -> None:
    exact(actual.value, expected["value"])
    assert actual.currency.code == expected["currency"]


def _eur_usd() -> ExchangeRate:
    return ExchangeRate(EURCurrency(), USDCurrency(), 1.1)


@pytest.mark.exact
def test_direct_rate(cpp: dict[str, Any]) -> None:
    _assert_rate(_eur_usd(), cpp["direct"])


@pytest.mark.exact
def test_exchange_from_source(cpp: dict[str, Any]) -> None:
    _assert_money(_eur_usd().exchange(Money(100.0, EURCurrency())), cpp["exchange_from_source"])


@pytest.mark.exact
def test_exchange_from_target(cpp: dict[str, Any]) -> None:
    # A Direct rate is symmetric: coming from the target it divides.
    _assert_money(_eur_usd().exchange(Money(110.0, USDCurrency())), cpp["exchange_from_target"])


def test_exchange_unrelated_currency_raises(cpp: dict[str, Any]) -> None:
    assert cpp["exchange_unrelated_currency"]["raises"] is True
    with pytest.raises(LibraryException):
        _eur_usd().exchange(Money(100.0, JPYCurrency()))


@pytest.mark.exact
def test_chain_source_eq_source(cpp: dict[str, Any]) -> None:
    other = ExchangeRate(EURCurrency(), GBPCurrency(), 0.85)
    _assert_rate(ExchangeRate.chain(_eur_usd(), other), cpp["chain_source_eq_source"])


@pytest.mark.exact
def test_chain_source_eq_target(cpp: dict[str, Any]) -> None:
    other = ExchangeRate(GBPCurrency(), EURCurrency(), 1.2)
    _assert_rate(ExchangeRate.chain(_eur_usd(), other), cpp["chain_source_eq_target"])


@pytest.mark.exact
def test_chain_target_eq_source(cpp: dict[str, Any]) -> None:
    other = ExchangeRate(USDCurrency(), JPYCurrency(), 150.0)
    _assert_rate(ExchangeRate.chain(_eur_usd(), other), cpp["chain_target_eq_source"])


@pytest.mark.exact
def test_chain_target_eq_target(cpp: dict[str, Any]) -> None:
    other = ExchangeRate(GBPCurrency(), USDCurrency(), 1.3)
    _assert_rate(ExchangeRate.chain(_eur_usd(), other), cpp["chain_target_eq_target"])


def test_chain_without_a_shared_currency_raises(cpp: dict[str, Any]) -> None:
    assert cpp["chain_not_chainable"]["raises"] is True
    with pytest.raises(LibraryException):
        ExchangeRate.chain(_eur_usd(), ExchangeRate(GBPCurrency(), JPYCurrency(), 2.0))


@pytest.mark.exact
def test_derived_exchange_from_source(cpp: dict[str, Any]) -> None:
    chained = ExchangeRate.chain(_eur_usd(), ExchangeRate(USDCurrency(), JPYCurrency(), 150.0))
    _assert_money(chained.exchange(Money(100.0, EURCurrency())), cpp["chain_exchange_from_source"])


@pytest.mark.exact
def test_derived_exchange_from_target(cpp: dict[str, Any]) -> None:
    chained = ExchangeRate.chain(_eur_usd(), ExchangeRate(USDCurrency(), JPYCurrency(), 150.0))
    _assert_money(chained.exchange(Money(16500.0, JPYCurrency())), cpp["chain_exchange_from_target"])


# --- API semantics not worth a probe case -------------------------------


def test_default_constructed_rate_is_an_unusable_placeholder() -> None:
    # C++ parity: the default ctor leaves rate_ at Null<Decimal>() and both
    # currencies empty; Python spells the null rate ``None``.
    placeholder = ExchangeRate()
    assert placeholder.rate is None
    assert placeholder.type == ExchangeRateType.Direct
    assert placeholder.source == Currency()
    assert placeholder.target == Currency()
    with pytest.raises(LibraryException):
        placeholder.exchange(Money(1.0, EURCurrency()))
