"""Tests for BasketOption + BasketPayoff hierarchy.

# C++ parity: ql/instruments/basketoption.{hpp,cpp} @ v1.42.1.
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.basket_option import (
    AverageBasketPayoff,
    BasketOption,
    MaxBasketPayoff,
    MinBasketPayoff,
    SpreadBasketPayoff,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.testing import tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _base_call() -> PlainVanillaPayoff:
    return PlainVanillaPayoff(OptionType.Call, 100.0)


def _expiry() -> Date:
    return Date.from_ymd(15, Month.June, 2027)


# --- MinBasketPayoff -------------------------------------------------------


def test_min_basket_accumulate() -> None:
    p = MinBasketPayoff(_base_call())
    tolerance.exact(p.accumulate([110.0, 95.0, 120.0]), 95.0)


def test_min_basket_evaluate_at_strike() -> None:
    """Call payoff on min(prices) = max(min - K, 0)."""
    p = MinBasketPayoff(_base_call())
    tolerance.exact(p.evaluate([110.0, 95.0, 120.0]), 0.0)  # min=95 < 100
    tolerance.exact(p.evaluate([110.0, 105.0, 120.0]), 5.0)  # min=105


def test_min_basket_empty_rejected() -> None:
    p = MinBasketPayoff(_base_call())
    with pytest.raises(LibraryException, match="empty price array"):
        p.accumulate([])


# --- MaxBasketPayoff -------------------------------------------------------


def test_max_basket_accumulate() -> None:
    p = MaxBasketPayoff(_base_call())
    tolerance.exact(p.accumulate([110.0, 95.0, 120.0]), 120.0)


def test_max_basket_evaluate_at_strike() -> None:
    """Call payoff on max(prices) = max(max - K, 0)."""
    p = MaxBasketPayoff(_base_call())
    tolerance.exact(p.evaluate([110.0, 95.0, 120.0]), 20.0)


# --- AverageBasketPayoff ---------------------------------------------------


def test_average_basket_uniform_weights() -> None:
    p = AverageBasketPayoff(_base_call(), n=3)
    tolerance.tight(p.accumulate([110.0, 95.0, 120.0]), (110.0 + 95.0 + 120.0) / 3.0)


def test_average_basket_custom_weights() -> None:
    p = AverageBasketPayoff(_base_call(), weights=[0.5, 0.3, 0.2])
    expected = 0.5 * 110.0 + 0.3 * 95.0 + 0.2 * 120.0
    tolerance.tight(p.accumulate([110.0, 95.0, 120.0]), expected)


def test_average_basket_rejects_shape_mismatch() -> None:
    p = AverageBasketPayoff(_base_call(), n=3)
    with pytest.raises(LibraryException, match="length mismatch"):
        p.accumulate([110.0, 95.0])  # length=2 vs weights=3


# --- SpreadBasketPayoff ----------------------------------------------------


def test_spread_basket_accumulate_two_assets() -> None:
    p = SpreadBasketPayoff(_base_call())
    tolerance.exact(p.accumulate([110.0, 100.0]), 10.0)


def test_spread_basket_rejects_wrong_count() -> None:
    p = SpreadBasketPayoff(_base_call())
    with pytest.raises(LibraryException, match="only defined for two"):
        p.accumulate([110.0, 95.0, 120.0])


# --- BasketOption itself --------------------------------------------------


def test_basket_option_holds_payoff_and_exercise() -> None:
    bp = MinBasketPayoff(_base_call())
    ex = EuropeanExercise(_expiry())
    opt = BasketOption(bp, ex)
    assert opt.payoff() is bp
    assert opt.exercise() is ex
    assert not opt.is_expired()
