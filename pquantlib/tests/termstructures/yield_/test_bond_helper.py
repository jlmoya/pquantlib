"""Tests for BondHelper. Full implied_quote test deferred to L3."""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.bond_helper import BondHelper, BondPriceType


class _StubBond:
    """Placeholder until L3 ports the Bond instrument."""

    name: str = "stub"


def _stub_bond() -> Any:
    return _StubBond()


def test_bond_helper_holds_bond_and_price_type() -> None:
    bond = _stub_bond()
    helper = BondHelper(SimpleQuote(100.0), bond)
    assert helper.bond() is bond
    assert helper.price_type() == BondPriceType.Clean


def test_bond_helper_dirty_price_type() -> None:
    helper = BondHelper(SimpleQuote(100.0), _stub_bond(), price_type=BondPriceType.Dirty)
    assert helper.price_type() == BondPriceType.Dirty


def test_bond_helper_implied_quote_deferred() -> None:
    helper = BondHelper(SimpleQuote(100.0), _stub_bond())
    with pytest.raises(LibraryException, match="L3"):
        helper.implied_quote()
