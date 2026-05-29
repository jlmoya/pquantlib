"""Unit tests for yield-curve bootstrap traits.

# C++ parity: ql/termstructures/yield/bootstraptraits.hpp.

Traits are pure static helpers; cross-check by reproducing the C++
``Discount`` / ``ZeroYield`` / ``ForwardRate`` arithmetic exactly.
"""

from __future__ import annotations

import math

from pquantlib.termstructures.yield_.yield_traits import (
    Discount,
    ForwardRate,
    ZeroYield,
)

# C++ parity: ql/termstructures/yield/bootstraptraits.hpp anon namespace.
_AVG_RATE = 0.05
_MAX_RATE = 1.0


# --- Discount --------------------------------------------------------------


def test_discount_initial_value_is_one() -> None:
    """C++ parity: ``Discount::initialValue`` = 1.0."""
    assert Discount().initial_value(None) == 1.0


def test_discount_first_pillar_guess() -> None:
    """C++ parity: first-pillar guess = 1 / (1 + avg_rate * 0.25)."""
    expected = 1.0 / (1.0 + _AVG_RATE * 0.25)
    assert Discount().guess(1, [1.0, 0.0], valid_data=False) == expected


def test_discount_guess_valid_data_reads_previous() -> None:
    assert Discount().guess(2, [1.0, 0.98, 0.95], valid_data=True) == 0.95


def test_discount_max_value_after_caps_at_previous() -> None:
    """Discount factors are monotonically decreasing."""
    assert Discount().max_value_after(2, [1.0, 0.98, 0.95], valid_data=False) == 0.98


def test_discount_update_guess_writes_pillar() -> None:
    data = [1.0, 0.0, 0.0]
    Discount().update_guess(data, 0.97, 1)
    assert data[1] == 0.97
    # discount: first point not propagated.
    assert data[0] == 1.0


def test_discount_max_iterations_is_100() -> None:
    assert Discount().max_iterations() == 100


# --- ZeroYield -------------------------------------------------------------


def test_zero_yield_initial_value_is_avg_rate() -> None:
    """C++ parity: ``ZeroYield::initialValue`` = avg_rate (dummy)."""
    assert ZeroYield().initial_value(None) == _AVG_RATE


def test_zero_yield_first_pillar_guess_is_avg_rate() -> None:
    assert ZeroYield().guess(1, [_AVG_RATE, 0.0], valid_data=False) == _AVG_RATE


def test_zero_yield_min_value_after_invalid_is_negative_max_rate() -> None:
    """No constraint → bracket starts at -max_rate."""
    assert ZeroYield().min_value_after(1, [], valid_data=False) == -_MAX_RATE


def test_zero_yield_max_value_after_invalid_is_max_rate() -> None:
    assert ZeroYield().max_value_after(1, [], valid_data=False) == _MAX_RATE


def test_zero_yield_min_after_valid_handles_negative_rates() -> None:
    """If min(data) < 0, double it; if > 0, halve it. C++ parity."""
    # Positive case: min is 0.02, expect 0.01.
    assert math.isclose(
        ZeroYield().min_value_after(1, [0.05, 0.04, 0.02], valid_data=True), 0.01,
    )
    # Negative case: min is -0.01, expect -0.02.
    assert math.isclose(
        ZeroYield().min_value_after(1, [0.05, 0.04, -0.01], valid_data=True), -0.02,
    )


def test_zero_yield_update_guess_propagates_to_first() -> None:
    """First pillar update propagates to t=0."""
    data = [_AVG_RATE, 0.0, 0.0]
    ZeroYield().update_guess(data, 0.025, 1)
    assert data[0] == 0.025
    assert data[1] == 0.025


# --- ForwardRate -----------------------------------------------------------


def test_forward_rate_initial_value_is_avg_rate() -> None:
    assert ForwardRate().initial_value(None) == _AVG_RATE


def test_forward_rate_update_guess_propagates_to_first() -> None:
    data = [_AVG_RATE, 0.0, 0.0]
    ForwardRate().update_guess(data, 0.035, 1)
    assert data[0] == 0.035
    assert data[1] == 0.035


def test_forward_rate_max_after_valid_handles_negative() -> None:
    """max(data) > 0 → r * 2.0; max(data) < 0 → r / 2.0."""
    assert math.isclose(
        ForwardRate().max_value_after(1, [0.05, 0.04], valid_data=True), 0.1,
    )
