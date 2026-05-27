"""Tests for pquantlib.termstructures.bootstrap_helper (BootstrapHelper + PillarChoice)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.bootstrap_helper import (
    BootstrapHelper,
    PillarChoice,
    RelativeDateBootstrapHelper,
)
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


# --- RelativeDateBootstrapHelper -----------------------------------------


class _StubRelativeHelper(RelativeDateBootstrapHelper[object]):
    """Concrete RelativeDateBootstrapHelper: latest date = eval_date + offset_days."""

    def __init__(self, quote: object, offset_days: int) -> None:
        self._offset_days = offset_days
        self._init_count = 0
        super().__init__(quote)  # type: ignore[arg-type]
        # Initial _initialize_dates was called after super().__init__()
        # set _eval_date_at_last_init = None; we need to also trigger an
        # initial date computation eagerly.
        self._initialize_dates()
        self._eval_date_at_last_init = ObservableSettings().evaluation_date_or_today()

    def _initialize_dates(self) -> None:
        today = ObservableSettings().evaluation_date_or_today()
        self._earliest_date = today
        self._latest_date = today + self._offset_days
        self._pillar_date = self._latest_date
        self._init_count += 1

    def implied_quote(self) -> float:
        return 0.0


@pytest.fixture
def clean_settings() -> Iterator[ObservableSettings]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    settings.evaluation_date = None
    yield settings
    settings.evaluation_date = None


def test_relative_date_helper_re_initializes_on_eval_date_change(
    clean_settings: ObservableSettings,
) -> None:
    """Moving eval_date must trigger _initialize_dates again."""
    clean_settings.evaluation_date = Date.from_ymd(15, Month.June, 2026)
    h = _StubRelativeHelper(SimpleQuote(0.05), offset_days=30)
    first_init = h._init_count  # pyright: ignore[reportPrivateUsage]
    first_latest = h.latest_date()
    clean_settings.evaluation_date = Date.from_ymd(15, Month.July, 2026)
    new_latest = h.latest_date()
    # update() must have been delivered and rebuilt dates.
    assert h._init_count > first_init  # pyright: ignore[reportPrivateUsage]
    assert new_latest != first_latest


def test_relative_date_helper_notifies_own_observers_on_eval_date_change(
    clean_settings: ObservableSettings,
) -> None:
    """RelativeDateBootstrapHelper forwards observer notifications."""
    clean_settings.evaluation_date = Date.from_ymd(15, Month.June, 2026)
    h = _StubRelativeHelper(SimpleQuote(0.05), offset_days=30)
    counts = [0]

    class _Counter:
        def update(self) -> None:
            counts[0] += 1

    obs = _Counter()
    h.register_with(obs)
    clean_settings.evaluation_date = Date.from_ymd(15, Month.July, 2026)
    assert counts[0] == 1
