"""Tests for pquantlib.quotes.simple_quote (cross-validated vs C++ probe)."""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact

_REF = reference_reader.load("quotes/foundations")


def test_simple_quote_returns_stored_value() -> None:
    q = SimpleQuote(5.0)
    exact(q.value(), _REF["simple_quote_value"])


def test_simple_quote_with_value_is_valid() -> None:
    q = SimpleQuote(5.0)
    assert q.is_valid() is _REF["simple_quote_is_valid"]


def test_default_constructed_simple_quote_is_invalid() -> None:
    q = SimpleQuote()
    assert q.is_valid() is _REF["simple_quote_default_is_valid"]


def test_invalid_simple_quote_raises_on_value() -> None:
    q = SimpleQuote()
    with pytest.raises(LibraryException, match="invalid SimpleQuote"):
        q.value()


def test_set_value_returns_diff() -> None:
    q = SimpleQuote(2.0)
    diff = q.set_value(7.0)
    exact(diff, _REF["simple_quote_set_value_diff"])
    exact(q.value(), _REF["simple_quote_after_set"])


def test_set_value_with_same_value_returns_zero_diff() -> None:
    q = SimpleQuote(3.0)
    exact(q.set_value(3.0), _REF["simple_quote_set_same_diff"])


def test_reset_makes_quote_invalid() -> None:
    q = SimpleQuote(4.0)
    q.reset()
    assert q.is_valid() is _REF["simple_quote_reset_is_valid"]


def test_set_value_notifies_observers_on_change() -> None:
    q = SimpleQuote(1.0)
    counts = [0]

    class _Counter:
        def update(self) -> None:
            counts[0] += 1

    obs = _Counter()
    q.register_with(obs)
    q.set_value(2.0)
    assert counts[0] == 1


def test_set_value_does_not_notify_observers_when_unchanged() -> None:
    q = SimpleQuote(1.0)
    counts = [0]

    class _Counter:
        def update(self) -> None:
            counts[0] += 1

    obs = _Counter()
    q.register_with(obs)
    q.set_value(1.0)
    assert counts[0] == 0
