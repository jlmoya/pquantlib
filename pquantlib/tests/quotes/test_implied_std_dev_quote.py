"""Tests for ImpliedStdDevQuote (cross-validated vs C++ v1.43).

# C++ parity: ql/quotes/impliedstddevquote.{hpp,cpp} @ v1.43.

Every expected value comes from
``migration-harness/cpp/probes/v143_quotes_tail/probe.cpp`` (section
``sectionImpliedStdDev``), captured in
``migration-harness/references/v143/quotes/tail.json``.
"""

from __future__ import annotations

from typing import Any, Final

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.payoffs import OptionType
from pquantlib.quotes.implied_std_dev_quote import ImpliedStdDevQuote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact, tight

_REF: Final[dict[str, Any]] = reference_reader.load("v143/quotes/tail")


def _quote(
    option_type: OptionType,
    forward: float | None,
    price: float | None,
    strike: float,
    guess: float,
    accuracy: float = 1.0e-10,
    max_iter: int = 100,
) -> ImpliedStdDevQuote:
    return ImpliedStdDevQuote(
        option_type,
        None if forward is None else SimpleQuote(forward),
        None if price is None else SimpleQuote(price),
        strike,
        guess,
        accuracy,
        max_iter,
    )


# --- converged inversions ----------------------------------------------------


@pytest.mark.parametrize(
    ("case", "option_type", "forward", "price", "strike", "guess"),
    [
        ("isdq_call_atm", OptionType.Call, 100.0, 5.0, 100.0, 0.15),
        ("isdq_put_itm", OptionType.Put, 100.0, 12.0, 110.0, 0.30),
        ("isdq_call_otm_rate", OptionType.Call, 0.05, 0.002, 0.06, 0.25),
        ("isdq_put_otm_rate", OptionType.Put, 0.05, 0.0015, 0.04, 0.25),
    ],
)
def test_value_inverts_the_black_formula(
    case: str,
    option_type: OptionType,
    forward: float,
    price: float,
    strike: float,
    guess: float,
) -> None:
    """Both option types, in and out of the money, at price and at rate scale."""
    quote = _quote(option_type, forward, price, strike, guess)
    tight(quote.value(), _REF[case])
    assert quote.is_valid() is _REF[case + "_is_valid"]


def test_trailing_arguments_default_to_the_cpp_values() -> None:
    """``accuracy=1e-6`` and ``max_iter=100``, reached through the defaults."""
    quote = ImpliedStdDevQuote(OptionType.Call, SimpleQuote(100.0), SimpleQuote(5.0), 100.0, 0.15)
    tight(quote.value(), _REF["isdq_defaults"])


# --- accuracy / guess / max_iter are threaded, not discarded -----------------


@pytest.mark.parametrize(
    ("case", "guess"),
    [("isdq_coarse_low_guess", 0.02), ("isdq_coarse_high_guess", 2.0)],
)
def test_a_coarse_accuracy_stops_the_solve_short_of_the_root(case: str, guess: float) -> None:
    """At ``accuracy=0.5`` the answer depends on the guess and is not the root.

    This is what pins ``accuracy`` and ``guess`` as live arguments: a port that
    accepted and then discarded either one would return the converged
    ``isdq_call_atm`` value for both rows.
    """
    quote = _quote(OptionType.Call, 100.0, 5.0, 100.0, guess, accuracy=0.5)
    tight(quote.value(), _REF[case])
    assert _REF[case] != _REF["isdq_call_atm"], "coarse solve must differ from the root"


def test_coarse_solves_from_different_guesses_disagree() -> None:
    assert _REF["isdq_coarse_low_guess"] != _REF["isdq_coarse_high_guess"]


def test_exhausting_max_iter_reads_as_zero_volatility() -> None:
    """One evaluation is not enough; the solver's Error is swallowed to 0.0."""
    quote = _quote(OptionType.Call, 100.0, 5.0, 100.0, 2.0, max_iter=1)
    exact(quote.value(), _REF["isdq_max_iter_1"])
    assert quote.is_valid() is _REF["isdq_max_iter_1_is_valid"]


def test_a_price_below_intrinsic_reads_as_zero_volatility() -> None:
    """No solution exists; ``performCalculations`` catches and returns 0.0."""
    quote = _quote(OptionType.Call, 100.0, 0.5, 90.0, 0.15)
    exact(quote.value(), _REF["isdq_price_below_intrinsic"])


# --- the previous result seeds the next solve --------------------------------


def test_a_recalculation_is_seeded_with_the_previous_result() -> None:
    """``impliedStdev_`` is mutable: the second solve starts where the first ended.

    Pinned at a coarse accuracy, where the starting point still shows in the
    answer — the re-solve after the price moves lands on a different number
    than a freshly-constructed quote given the same inputs.
    """
    price = SimpleQuote(5.0)
    quote = ImpliedStdDevQuote(
        OptionType.Call, SimpleQuote(100.0), price, 100.0, 2.0, 0.5, 100
    )
    tight(quote.value(), _REF["isdq_restart_first"])
    price.set_value(9.0)
    tight(quote.value(), _REF["isdq_restart_second"])

    fresh = _quote(OptionType.Call, 100.0, 9.0, 100.0, 2.0, accuracy=0.5)
    tight(fresh.value(), _REF["isdq_restart_fresh"])
    assert _REF["isdq_restart_second"] != _REF["isdq_restart_fresh"]


# --- validity and empty handles ----------------------------------------------


@pytest.mark.parametrize(
    ("case", "forward", "price"),
    [
        ("isdq_invalid_price_is_valid", SimpleQuote(100.0), SimpleQuote()),
        ("isdq_invalid_forward_is_valid", SimpleQuote(), SimpleQuote(5.0)),
    ],
)
def test_an_invalid_input_quote_makes_the_quote_invalid(
    case: str, forward: SimpleQuote, price: SimpleQuote
) -> None:
    quote = ImpliedStdDevQuote(OptionType.Call, forward, price, 100.0, 0.15, 1.0e-10, 100)
    assert quote.is_valid() is _REF[case]


def test_a_missing_forward_is_swallowed_but_a_missing_price_is_not() -> None:
    """The C++ ``try`` block starts *after* ``price_->value()``.

    ``forward_->value()`` is inside it, so an empty forward handle reads as
    0.0; an empty price handle raises. A port that wrapped the whole body in
    one ``try`` would return 0.0 for both.
    """
    no_forward = _quote(OptionType.Call, None, 5.0, 100.0, 0.15)
    assert no_forward.is_valid() is _REF["isdq_empty_forward_is_valid"]
    exact(no_forward.value(), _REF["isdq_empty_forward_value"])

    no_price = _quote(OptionType.Call, 100.0, None, 100.0, 0.15)
    assert no_price.is_valid() is _REF["isdq_empty_price_is_valid"]
    assert _REF["isdq_empty_price_value_raises"] is True
    with pytest.raises(LibraryException, match="empty Handle cannot be dereferenced"):
        no_price.value()


# --- lazy-object plumbing ----------------------------------------------------


def test_the_quote_notifies_its_own_observers_when_an_input_moves() -> None:
    forward = SimpleQuote(100.0)
    quote = ImpliedStdDevQuote(OptionType.Call, forward, SimpleQuote(5.0), 100.0, 0.15)
    counts = [0]

    class _Counter:
        def update(self) -> None:
            counts[0] += 1

    observer = _Counter()
    quote.register_with(observer)
    forward.set_value(101.0)
    assert counts[0] == 1
