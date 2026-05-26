"""Cross-validate ISO 4217 currency descriptors against the C++ probe.

Probe key: cluster/b -> "currencies".

The C++ probe emits the ISO codes for USD/EUR/GBP/JPY/CHF plus USD's
full name, symbol, numeric code, and fractions-per-unit. We assert the
same attribute set on each Python currency class.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.currencies.asia import JPYCurrency
from pquantlib.currencies.currency import Currency
from pquantlib.currencies.europe import CHFCurrency, EURCurrency, GBPCurrency
from pquantlib.testing import reference_reader


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/b")


def test_usd_code(cpp: dict[str, Any]) -> None:
    assert USDCurrency().code == cpp["currencies"]["USD"]


def test_eur_code(cpp: dict[str, Any]) -> None:
    assert EURCurrency().code == cpp["currencies"]["EUR"]


def test_gbp_code(cpp: dict[str, Any]) -> None:
    assert GBPCurrency().code == cpp["currencies"]["GBP"]


def test_jpy_code(cpp: dict[str, Any]) -> None:
    assert JPYCurrency().code == cpp["currencies"]["JPY"]


def test_chf_code(cpp: dict[str, Any]) -> None:
    assert CHFCurrency().code == cpp["currencies"]["CHF"]


def test_usd_full_attributes(cpp: dict[str, Any]) -> None:
    usd = USDCurrency()
    assert usd.name == cpp["currencies"]["USD_name"]
    assert usd.symbol == cpp["currencies"]["USD_symbol"]
    assert usd.numeric_code == int(cpp["currencies"]["USD_numeric"])
    assert usd.fractions_per_unit == int(cpp["currencies"]["USD_fractions"])


# --- Currency semantics -------------------------------------------------


def test_currency_equality_by_name() -> None:
    # C++ ``operator==`` compares Currency by ``name`` only. Two distinct
    # Python instances of the same currency class compare equal.
    assert USDCurrency() == USDCurrency()
    assert hash(USDCurrency()) == hash(USDCurrency())


def test_currencies_with_different_names_are_unequal() -> None:
    assert USDCurrency() != EURCurrency()


def test_default_currency_is_empty() -> None:
    c = Currency()
    assert c.empty()
    assert c.code == ""
    assert c.name == ""


def test_empty_currencies_are_equal() -> None:
    assert Currency() == Currency()


def test_empty_vs_non_empty_unequal() -> None:
    assert Currency() != USDCurrency()


def test_str_returns_code() -> None:
    assert str(USDCurrency()) == "USD"


def test_str_empty_currency() -> None:
    assert str(Currency()) == "(null currency)"
