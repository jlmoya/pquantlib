"""Tests for MultiCompositeQuote (cross-validated vs C++ v1.43).

# C++ parity: ql/quotes/multicompositequote.hpp @ v1.43.

Every expected value comes from
``migration-harness/cpp/probes/v143_quotes_tail/probe.cpp`` (section
``sectionMultiComposite``), captured in
``migration-harness/references/v143/quotes/tail.json``.
"""

from __future__ import annotations

import math
from typing import Any, Final

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.quotes.multi_composite_quote import MultiCompositeQuote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact, tight

_REF: Final[dict[str, Any]] = reference_reader.load("v143/quotes/tail")

ELEMENT_VALUES: Final[tuple[float, float, float]] = (2.0, 3.0, 5.0)


def _weighted(a: Array) -> float:
    """Probe.cpp:566 — a distinct coefficient per slot, so ORDER is pinned."""
    return float(a[0] + 10.0 * a[1] + 100.0 * a[2])


def _norm(a: Array) -> float:
    """Probe.cpp:590-595 — reads the whole Array, not just the first slots."""
    return float(math.sqrt(sum(float(x) * float(x) for x in a)))


def _elements() -> list[SimpleQuote]:
    return [SimpleQuote(v) for v in ELEMENT_VALUES]


def test_value_applies_the_function_to_the_elements_in_order() -> None:
    quote = MultiCompositeQuote(_elements(), _weighted)
    exact(quote.value(), _REF["mcq_weighted_value"])
    assert quote.is_valid() is _REF["mcq_weighted_is_valid"]


def test_the_function_receives_every_element() -> None:
    quote = MultiCompositeQuote(_elements(), _norm)
    tight(quote.value(), _REF["mcq_norm_value"])
    assert _REF["mcq_norm_size"] == len(ELEMENT_VALUES)


def test_input_value_reads_each_element() -> None:
    quote = MultiCompositeQuote(_elements(), _weighted)
    exact(quote.input_value(0), _REF["mcq_input_value_0"])
    exact(quote.input_value(1), _REF["mcq_input_value_1"])
    exact(quote.input_value(2), _REF["mcq_input_value_2"])


def test_input_value_is_bounds_checked() -> None:
    """C++ uses ``elements_.at(i)``; Python's list indexing raises IndexError."""
    assert _REF["mcq_input_value_3_raises"] is True
    with pytest.raises(IndexError):
        MultiCompositeQuote(_elements(), _weighted).input_value(3)


def test_an_element_moving_drops_the_cached_value() -> None:
    elements = _elements()
    quote = MultiCompositeQuote(elements, _weighted)
    exact(quote.value(), _REF["mcq_weighted_value"])
    elements[1].set_value(4.0)
    exact(quote.value(), _REF["mcq_after_element_moves"])


def test_an_invalid_element_makes_the_quote_invalid() -> None:
    elements: list[SimpleQuote | None] = [SimpleQuote(2.0), SimpleQuote(), SimpleQuote(5.0)]
    quote = MultiCompositeQuote(elements, _weighted)
    assert quote.is_valid() is _REF["mcq_invalid_element_is_valid"]
    assert _REF["mcq_invalid_element_value_raises"] is True
    with pytest.raises(LibraryException, match="invalid MultiCompositeQuote"):
        quote.value()


def test_an_empty_element_handle_makes_the_quote_invalid() -> None:
    """``None`` models C++'s empty ``Handle<Quote>`` in the element vector."""
    elements: list[SimpleQuote | None] = [SimpleQuote(2.0), None, SimpleQuote(5.0)]
    quote = MultiCompositeQuote(elements, _weighted)
    assert quote.is_valid() is _REF["mcq_empty_element_is_valid"]
    assert _REF["mcq_empty_element_value_raises"] is True
    with pytest.raises(LibraryException, match="invalid MultiCompositeQuote"):
        quote.value()


def test_no_elements_is_vacuously_valid() -> None:
    quote = MultiCompositeQuote([], lambda a: float(a.size))
    assert quote.is_valid() is _REF["mcq_no_elements_is_valid"]
    exact(quote.value(), _REF["mcq_no_elements_value"])


def test_the_function_is_handed_a_float64_array() -> None:
    """C++'s ``ArrayFunction`` takes a ``QuantLib::Array``; PQuantLib's is numpy."""
    seen: list[Array] = []

    def _capture(a: Array) -> float:
        seen.append(a)
        return 0.0

    MultiCompositeQuote(_elements(), _capture).value()
    assert len(seen) == 1
    assert seen[0].dtype == np.float64
    assert seen[0].tolist() == list(ELEMENT_VALUES)


def test_the_quote_notifies_its_own_observers_when_an_element_moves() -> None:
    elements = _elements()
    quote = MultiCompositeQuote(elements, _weighted)
    counts = [0]

    class _Counter:
        def update(self) -> None:
            counts[0] += 1

    observer = _Counter()
    quote.register_with(observer)
    elements[0].set_value(9.0)
    assert counts[0] == 1
