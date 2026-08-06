"""Tests for EurodollarFuturesImpliedStdDevQuote (cross-validated vs C++ v1.43).

# C++ parity: ql/quotes/eurodollarfuturesquote.{hpp,cpp} @ v1.43.

Every expected value comes from
``migration-harness/cpp/probes/v143_quotes_tail/probe.cpp`` (section
``sectionEurodollar``), captured in
``migration-harness/references/v143/quotes/tail.json``.

Setup shared with the probe: the forward is a futures price of 94.85 (a rate
forward of 5.15), the call quote is 0.25 and the put quote is 0.30 — chosen
deliberately different so a port that reached for the wrong one on either
branch cannot reproduce the numbers.
"""

from __future__ import annotations

from typing import Any, Final

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.quotes.eurodollar_futures_quote import EurodollarFuturesImpliedStdDevQuote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight

_REF: Final[dict[str, Any]] = reference_reader.load("v143/quotes/tail")

FORWARD_PRICE: Final[float] = 94.85
CALL_PRICE: Final[float] = 0.25
PUT_PRICE: Final[float] = 0.30

# strike 94.0 -> rate strike 6.0, above the 5.15 rate forward -> "put branch".
# strike 95.0 -> rate strike 5.0, below it                    -> "call branch".
PUT_BRANCH_STRIKE: Final[float] = 94.0
CALL_BRANCH_STRIKE: Final[float] = 95.0


def _maybe(value: float | None) -> SimpleQuote | None:
    """``None`` models C++'s empty ``Handle<Quote>``; a float, a populated one."""
    return None if value is None else SimpleQuote(value)


def _quote(
    strike: float,
    guess: float = 0.15,
    accuracy: float = 1.0e-10,
    max_iter: int = 100,
    *,
    forward: float | None = FORWARD_PRICE,
    call_price: float | None = CALL_PRICE,
    put_price: float | None = PUT_PRICE,
) -> EurodollarFuturesImpliedStdDevQuote:
    return EurodollarFuturesImpliedStdDevQuote(
        _maybe(forward),
        _maybe(call_price),
        _maybe(put_price),
        strike,
        guess,
        accuracy,
        max_iter,
    )


# --- both branches of the price/rate inversion -------------------------------


def test_a_rate_strike_above_the_forward_prices_a_call_off_the_put_quote() -> None:
    quote = _quote(PUT_BRANCH_STRIKE)
    tight(quote.value(), _REF["edf_put_branch"])
    assert quote.is_valid() is _REF["edf_put_branch_is_valid"]


def test_a_rate_strike_below_the_forward_prices_a_put_off_the_call_quote() -> None:
    quote = _quote(CALL_BRANCH_STRIKE)
    tight(quote.value(), _REF["edf_call_branch"])
    assert quote.is_valid() is _REF["edf_call_branch_is_valid"]


def test_the_two_branches_disagree() -> None:
    """Guards the parametrised cases above: the branch really is observable."""
    assert _REF["edf_put_branch"] != _REF["edf_call_branch"]


# --- trailing arguments are threaded, not discarded --------------------------


def test_trailing_arguments_default_to_the_cpp_values() -> None:
    """``guess=0.15``, ``accuracy=1e-6``, ``max_iter=100`` through the defaults."""
    quote = EurodollarFuturesImpliedStdDevQuote(
        SimpleQuote(FORWARD_PRICE),
        SimpleQuote(CALL_PRICE),
        SimpleQuote(PUT_PRICE),
        CALL_BRANCH_STRIKE,
    )
    tight(quote.value(), _REF["edf_defaults"])


def test_a_coarse_accuracy_stops_the_solve_short_of_the_root() -> None:
    quote = _quote(CALL_BRANCH_STRIKE, guess=2.0, accuracy=0.5)
    tight(quote.value(), _REF["edf_coarse_high_guess"])
    assert _REF["edf_coarse_high_guess"] != _REF["edf_call_branch"]


def test_exhausting_max_iter_propagates() -> None:
    """Unlike ImpliedStdDevQuote this class does NOT catch the solver's Error."""
    assert _REF["edf_max_iter_1_raises"] is True
    with pytest.raises(LibraryException):
        _quote(CALL_BRANCH_STRIKE, guess=2.0, max_iter=1).value()


# --- validity: only the branch-relevant price handle matters -----------------


@pytest.mark.parametrize(
    ("case", "strike", "call_price", "put_price"),
    [
        ("edf_put_branch_empty_call_is_valid", PUT_BRANCH_STRIKE, None, PUT_PRICE),
        ("edf_put_branch_empty_put_is_valid", PUT_BRANCH_STRIKE, CALL_PRICE, None),
        ("edf_call_branch_empty_put_is_valid", CALL_BRANCH_STRIKE, CALL_PRICE, None),
        ("edf_call_branch_empty_call_is_valid", CALL_BRANCH_STRIKE, None, PUT_PRICE),
    ],
)
def test_is_valid_only_consults_the_branch_relevant_price(
    case: str, strike: float, call_price: float | None, put_price: float | None
) -> None:
    quote = _quote(strike, call_price=call_price, put_price=put_price)
    assert quote.is_valid() is _REF[case]


def test_a_missing_forward_makes_the_quote_invalid() -> None:
    assert _quote(CALL_BRANCH_STRIKE, forward=None).is_valid() is _REF["edf_empty_forward_is_valid"]


def test_an_invalid_forward_makes_the_quote_invalid() -> None:
    quote = EurodollarFuturesImpliedStdDevQuote(
        SimpleQuote(),  # populated handle, but the quote itself has no value
        SimpleQuote(CALL_PRICE),
        SimpleQuote(PUT_PRICE),
        CALL_BRANCH_STRIKE,
        0.15,
        1.0e-10,
        100,
    )
    assert quote.is_valid() is _REF["edf_invalid_forward_is_valid"]
