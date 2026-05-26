"""Tests for pquantlib.termstructures.bootstrap_helper (BootstrapHelper + PillarChoice)."""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper, PillarChoice
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class _StubHelper(BootstrapHelper[object]):
    """Minimal concrete BootstrapHelper for behavior tests."""

    def __init__(
        self,
        quote: object,
        *,
        earliest: Date | None = None,
        latest: Date | None = None,
        pillar: Date | None = None,
        maturity: Date | None = None,
        relevant: Date | None = None,
        implied: float = 0.0,
    ) -> None:
        super().__init__(quote)  # type: ignore[arg-type]
        self._earliest_date = earliest
        self._latest_date = latest
        self._pillar_date = pillar
        self._maturity_date = maturity
        self._latest_relevant_date = relevant
        self._implied = implied

    def implied_quote(self) -> float:
        return self._implied


# --- PillarChoice -----------------------------------------------------------


def test_pillar_choice_string_repr_matches_cpp() -> None:
    assert str(PillarChoice.MaturityDate) == "MaturityPillarDate"
    assert str(PillarChoice.LastRelevantDate) == "LastRelevantPillarDate"
    assert str(PillarChoice.CustomDate) == "CustomPillarDate"


# --- BootstrapHelper --------------------------------------------------------


def test_cannot_instantiate_abstract_helper() -> None:
    with pytest.raises(TypeError):
        BootstrapHelper(SimpleQuote(0.05))  # type: ignore[abstract]


def test_quote_error_is_market_minus_implied() -> None:
    q = SimpleQuote(0.05)
    h = _StubHelper(q, pillar=Date.from_ymd(1, Month.January, 2027), implied=0.045)
    tight(h.quote_error(), 0.005)


def test_set_term_structure_rejects_none() -> None:
    h = _StubHelper(SimpleQuote(0.05), pillar=Date.from_ymd(1, Month.January, 2027))
    with pytest.raises(LibraryException, match="null term structure"):
        h.set_term_structure(None)  # type: ignore[arg-type]


def test_set_term_structure_stores_reference() -> None:
    h = _StubHelper(SimpleQuote(0.05), pillar=Date.from_ymd(1, Month.January, 2027))
    sentinel = object()
    h.set_term_structure(sentinel)
    assert h._term_structure is sentinel  # pyright: ignore[reportPrivateUsage]


def test_date_fallback_chain_pillar_to_latest() -> None:
    # Only latest_date is set; pillar / maturity / relevant fall through to it.
    latest = Date.from_ymd(1, Month.July, 2027)
    h = _StubHelper(SimpleQuote(0.05), latest=latest, pillar=latest)
    assert h.latest_date() == latest
    assert h.pillar_date() == latest
    assert h.latest_relevant_date() == latest
    assert h.maturity_date() == latest


def test_explicit_dates_override_fallback() -> None:
    pillar = Date.from_ymd(1, Month.July, 2027)
    maturity = Date.from_ymd(15, Month.July, 2027)
    h = _StubHelper(SimpleQuote(0.05), pillar=pillar, maturity=maturity)
    assert h.pillar_date() == pillar
    assert h.maturity_date() == maturity


def test_helper_relays_observer_updates_from_quote() -> None:
    q = SimpleQuote(0.05)
    h = _StubHelper(q, pillar=Date.from_ymd(1, Month.July, 2027))
    counts = [0]

    class _Counter:
        def update(self) -> None:
            counts[0] += 1

    obs = _Counter()
    h.register_with(obs)
    q.set_value(0.06)  # triggers q.notify_observers → h.update → h.notify_observers
    assert counts[0] == 1


def test_helper_wraps_float_quote_in_simple_quote() -> None:
    h = _StubHelper(0.05, pillar=Date.from_ymd(1, Month.July, 2027))  # type: ignore[arg-type]
    assert isinstance(h.quote(), SimpleQuote)
    assert h.quote().value() == 0.05
