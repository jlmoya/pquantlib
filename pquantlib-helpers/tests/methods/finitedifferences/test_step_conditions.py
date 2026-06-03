"""Unit tests for the legacy FD step conditions (WS3-FD2).

# Retired-API compat layer — see package docstring.

``AmericanCondition`` applies ``max(value, intrinsic)`` elementwise; this is
checked against a hand-built payoff/array. ``NullCondition`` is a no-op.
``StepConditionSet`` dispatches each member to the matching component array.
"""

from __future__ import annotations

import numpy as np

from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.testing import tolerance
from pquantlib_helpers.methods.finitedifferences.step_condition import (
    AmericanCondition,
    NullCondition,
    StepConditionSet,
)


def test_null_condition_is_noop() -> None:
    a = np.array([1.0, 2.0, 3.0])
    NullCondition().apply_to(a, 0.5)
    assert np.array_equal(a, np.array([1.0, 2.0, 3.0]))


def test_american_condition_from_array_takes_elementwise_max() -> None:
    # intrinsic comes from a fixed reference array
    intrinsic = np.array([5.0, 5.0, 5.0, 5.0])
    values = np.array([3.0, 6.0, 5.0, 1.0])
    AmericanCondition(values=intrinsic).apply_to(values, 0.0)
    # max(value, intrinsic) elementwise
    assert np.array_equal(values, np.array([5.0, 6.0, 5.0, 5.0]))


def test_american_condition_from_payoff_uses_plain_vanilla_intrinsic() -> None:
    # Put, strike 10: intrinsic = max(K - S, 0). Underlying grid = the values.
    strike = 10.0
    grid = np.array([6.0, 8.0, 12.0])
    payoff = PlainVanillaPayoff(OptionType.Put, strike)
    # Build the condition over the grid (intrinsic = payoff(grid[i]))
    cond = AmericanCondition(option_type=OptionType.Put, strike=strike)
    # Start the continuation values below intrinsic so the floor binds.
    values = grid.copy()
    cond.apply_to(values, 0.0)
    expected = np.array([max(grid[i], payoff(float(grid[i]))) for i in range(len(grid))])
    for got, want in zip(values, expected, strict=True):
        tolerance.exact(float(got), float(want))


def test_american_condition_requires_construction_args() -> None:
    try:
        AmericanCondition()
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for empty AmericanCondition()")


def test_step_condition_set_dispatches_per_component_in_order() -> None:
    cond_set: StepConditionSet = StepConditionSet()
    a0 = np.array([3.0, 6.0])
    a1 = np.array([1.0, 2.0])
    cond_set.push_back(AmericanCondition(values=np.array([5.0, 5.0])))
    cond_set.push_back(NullCondition())
    cond_set.apply_to([a0, a1], 0.0)
    # component 0 floored at 5; component 1 untouched
    assert np.array_equal(a0, np.array([5.0, 6.0]))
    assert np.array_equal(a1, np.array([1.0, 2.0]))
