"""Tests for pquantlib.quotes.composite_quote (cross-validated vs C++ probe)."""

from __future__ import annotations

import math

from pquantlib.quotes.composite_quote import CompositeQuote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact

_REF = reference_reader.load("quotes/foundations")


def test_composite_quote_applies_sum_function() -> None:
    a = SimpleQuote(3.0)
    b = SimpleQuote(4.0)
    cq = CompositeQuote(a, b, lambda x, y: x + y)
    exact(cq.value(), _REF["composite_quote_sum"])


def test_composite_quote_applies_hypot_function() -> None:
    a = SimpleQuote(3.0)
    b = SimpleQuote(4.0)
    cq = CompositeQuote(a, b, lambda x, y: math.sqrt(x * x + y * y))
    exact(cq.value(), _REF["composite_quote_hypot"])


def test_composite_quote_exposes_value1_and_value2() -> None:
    a = SimpleQuote(3.0)
    b = SimpleQuote(4.0)
    cq = CompositeQuote(a, b, lambda x, y: math.sqrt(x * x + y * y))
    exact(cq.value1(), _REF["composite_quote_value1"])
    exact(cq.value2(), _REF["composite_quote_value2"])


def test_composite_quote_invalid_when_either_element_invalid() -> None:
    a = SimpleQuote(3.0)
    b = SimpleQuote()
    cq = CompositeQuote(a, b, lambda x, y: x + y)
    assert cq.is_valid() is False
