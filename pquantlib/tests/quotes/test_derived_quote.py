"""Tests for pquantlib.quotes.derived_quote (cross-validated vs C++ probe)."""

from __future__ import annotations

from pquantlib.quotes.derived_quote import DerivedQuote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact

_REF = reference_reader.load("quotes/foundations")


def test_derived_quote_applies_linear_function() -> None:
    underlying = SimpleQuote(3.0)
    dq = DerivedQuote(underlying, lambda x: 2.0 * x + 1.0)
    exact(dq.value(), _REF["derived_quote_linear"])


def test_derived_quote_applies_square_function() -> None:
    underlying = SimpleQuote(4.0)
    dq = DerivedQuote(underlying, lambda x: x * x)
    exact(dq.value(), _REF["derived_quote_square"])


def test_derived_quote_recomputes_on_underlying_change() -> None:
    underlying = SimpleQuote(3.0)
    dq = DerivedQuote(underlying, lambda x: 2.0 * x + 1.0)
    # Touch once to populate cache.
    assert dq.value() == 7.0
    # Change underlying → derived recomputes on next read.
    underlying.set_value(10.0)
    exact(dq.value(), _REF["derived_quote_after_update"])


def test_derived_quote_invalid_when_underlying_invalid() -> None:
    underlying = SimpleQuote()
    dq = DerivedQuote(underlying, lambda x: x + 1.0)
    assert dq.is_valid() is False
