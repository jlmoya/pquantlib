"""Gsr's quote-driven volatility and reversion, i.e. its two Observer adapters.

Every expected value comes from
``migration-harness/cpp/probes/v143_models_gsr_quotes/probe.cpp`` run against
C++ QuantLib v1.43; the reference lives at
``migration-harness/references/v143/models/gsr_quotes.json``.

``Gsr::VolatilityObserver`` (gsr.hpp:176) and ``Gsr::ReversionObserver``
(gsr.hpp:181) exist because one C++ class cannot implement ``Observer::update()``
twice; each routes a quote's notification to a different method,
``Gsr::updateVolatility`` / ``Gsr::updateReversion`` (gsr.cpp:121-135). All four
C++ constructors store ``Handle<Quote>``; the plain-``Real`` overloads simply
wrap each number in a ``SimpleQuote`` (gsr.cpp:36-40, :57-62), so observability
is unconditional rather than a property of the "quote" constructors.

The probe sets ``Settings::instance().evaluationDate() = TODAY`` as its first
statement in ``main``; the ``_pinned_evaluation_date`` autouse fixture pins the
same date (read from the reference's ``today_serial``) and restores it in
teardown.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.models.shortrate.onefactor.gsr import Gsr, ReversionObserver, VolatilityObserver
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date

_REF: Final[dict[str, Any]] = load_reference("v143/models/gsr_quotes")
_TODAY: Final[Date] = Date(int(_REF["today_serial"]))
_VOLSTEPDATES: Final[list[Date]] = [Date(int(s)) for s in _REF["volstepdate_serials"]]


@pytest.fixture(autouse=True)
def _pinned_evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = _TODAY
    try:
        yield
    finally:
        settings.evaluation_date = previous


def _curve() -> FlatForward:
    return FlatForward(_TODAY, SimpleQuote(0.03), Actual365Fixed())


def _assert_state(model: Gsr, expected: dict[str, Any], label: str) -> None:
    assert list(model.volatility()) == pytest.approx(expected["volatility"], abs=0.0, rel=1e-12), (
        label
    )
    assert list(model.reversion()) == pytest.approx(expected["reversion"], abs=0.0, rel=1e-12), (
        label
    )
    tight(model.zerobond(5.0, 1.0, 0.5), expected["zerobond_5_1_0p5"], reason=label)
    tight(model.numeraire(1.0, 0.5), expected["numeraire_1_0p5"], reason=label)


def test_single_reversion_quote_bumps_propagate() -> None:
    """Bumping a quote must reach the Parameters AND the GsrProcess cache."""
    section = _REF["single_reversion"]
    vol_quotes = [SimpleQuote(v) for v in (0.010, 0.012, 0.014)]
    rev_quote = SimpleQuote(0.01)
    model = Gsr(
        term_structure=_curve(),
        volstepdates=_VOLSTEPDATES,
        volatilities=vol_quotes,
        reversion=rev_quote,
        T=60.0,
    )
    _assert_state(model, section["before"], "before")

    # A port that stored the numbers instead of the quotes reproduces "before"
    # exactly and gets every "after" wrong — which is the point of reading the
    # state on both sides of the bump.
    vol_quotes[1].set_value(0.030)
    _assert_state(model, section["after_vol1_bump"], "after_vol1_bump")

    rev_quote.set_value(0.05)
    _assert_state(model, section["after_reversion_bump"], "after_reversion_bump")


def test_piecewise_reversion_quotes_bump_the_right_element() -> None:
    """ReversionObserver is registered against EVERY reversion quote."""
    section = _REF["piecewise_reversion"]
    vol_quotes = [SimpleQuote(v) for v in (0.010, 0.012, 0.014)]
    rev_quotes = [SimpleQuote(r) for r in (0.010, 0.020, 0.030)]
    model = Gsr(
        term_structure=_curve(),
        volstepdates=_VOLSTEPDATES,
        volatilities=vol_quotes,
        reversion=rev_quotes,
        T=60.0,
    )
    _assert_state(model, section["before"], "before")

    rev_quotes[2].set_value(0.075)
    _assert_state(model, section["after_reversion2_bump"], "after_reversion2_bump")


def test_plain_floats_are_wrapped_in_quotes() -> None:
    """C++ gsr.cpp:36-40 — the Real-taking ctors wrap each number in a SimpleQuote.

    So a float-built model is observably identical to a quote-built one, and the
    two observers are attached either way.
    """
    section = _REF["single_reversion"]["before"]
    model = Gsr(
        term_structure=_curve(),
        volstepdates=_VOLSTEPDATES,
        volatilities=[0.010, 0.012, 0.014],
        reversion=0.01,
        T=60.0,
    )
    _assert_state(model, section, "float ctor")


def test_the_two_observers_are_distinct_objects() -> None:
    """C++ needs two, because one class cannot implement Observer::update() twice."""
    model = Gsr(
        term_structure=_curve(),
        volstepdates=_VOLSTEPDATES,
        volatilities=[0.010, 0.012, 0.014],
        reversion=0.01,
        T=60.0,
    )
    vol_obs = model.volatility_observer()
    rev_obs = model.reversion_observer()
    assert isinstance(vol_obs, VolatilityObserver)
    assert isinstance(rev_obs, ReversionObserver)
    assert vol_obs is not rev_obs
