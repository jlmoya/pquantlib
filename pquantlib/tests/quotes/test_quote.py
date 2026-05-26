"""Tests for pquantlib.quotes.quote (Quote abstract base)."""

from __future__ import annotations

import pytest

from pquantlib.patterns.observer import Observable
from pquantlib.quotes.quote import Quote


def test_quote_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Quote()  # type: ignore[abstract]


def test_quote_is_an_observable() -> None:
    # Structural — every Quote inherits Observable's observer plumbing.
    assert issubclass(Quote, Observable)
