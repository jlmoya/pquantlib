"""Tests for the step-condition composite + FdmAmericanStepCondition.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.{hpp,cpp},
# ql/methods/finitedifferences/stepconditions/fdmamericanstepcondition.{hpp,cpp}
# @ v1.42.1.
"""

from __future__ import annotations

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.methods.finitedifferences.meshers.uniform_grid_mesher import (
    UniformGridMesher,
)
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpLayout,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_american_step_condition import (
    FdmAmericanStepCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.testing.tolerance import tight


def _build_log_spot_mesh() -> UniformGridMesher:
    """Build a mesh whose locations are in *log spot* space."""
    layout = FdmLinearOpLayout((11,))
    # log(50) ~ 3.91, log(150) ~ 5.01 → mesh in log-space covering ~spot=50..150.
    return UniformGridMesher(layout, [(np.log(50.0), np.log(150.0))])


def test_american_put_floor_replaces_when_payoff_higher() -> None:
    """American Put with K=100 — at log-spot below log(K), the payoff
    K - S is positive; if the FD value is below that, apply_to should
    floor it to the payoff.
    """
    mesh = _build_log_spot_mesh()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    cond = FdmAmericanStepCondition(mesh, payoff)
    # Initial FD values: all zero (below payoff at low-S nodes).
    n = mesh.layout().size()
    a = np.zeros(n, dtype=np.float64)
    cond.apply_to(a, 0.5)
    # All low-S nodes should now equal max(K - S, 0) > 0.
    locs = mesh.locations(0)
    for i, x in enumerate(locs):
        s = float(np.exp(x))
        expected = max(100.0 - s, 0.0)
        tight(float(a[i]), expected)


def test_american_no_floor_when_a_already_higher() -> None:
    """If a[i] already exceeds the payoff, apply_to leaves it alone."""
    mesh = _build_log_spot_mesh()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    cond = FdmAmericanStepCondition(mesh, payoff)
    n = mesh.layout().size()
    a = np.full(n, 200.0, dtype=np.float64)  # uniformly above payoff
    cond.apply_to(a, 0.5)
    # Unchanged.
    for v in a:
        tight(float(v), 200.0)


def test_american_respects_exercise_start() -> None:
    """If t < exercise_start, apply_to is a no-op."""
    mesh = _build_log_spot_mesh()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    cond = FdmAmericanStepCondition(mesh, payoff, exercise_start=0.5)
    n = mesh.layout().size()
    a = np.zeros(n, dtype=np.float64)
    cond.apply_to(a, 0.1)  # before exercise start
    # No flooring applied.
    for v in a:
        tight(float(v), 0.0)


def test_inconsistent_array_dimensions_raises() -> None:
    """apply_to with a wrong-length array must raise."""
    mesh = _build_log_spot_mesh()
    payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    cond = FdmAmericanStepCondition(mesh, payoff)
    bad = np.zeros(5, dtype=np.float64)
    with pytest.raises(LibraryException):
        cond.apply_to(bad, 0.5)


def test_composite_empty_is_no_op() -> None:
    """Composite with no conditions doesn't touch the array."""
    composite = FdmStepConditionComposite([], [])
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    composite.apply_to(a, 1.0)
    for actual, expected in zip(a, [1.0, 2.0, 3.0], strict=True):
        tight(float(actual), expected)


def test_composite_stopping_times_dedup_and_sort() -> None:
    """Stopping times across conditions get deduped and sorted."""
    composite = FdmStepConditionComposite([[1.0, 0.5], [0.5, 2.0]], [])
    assert composite.stopping_times() == [0.5, 1.0, 2.0]


def test_composite_applies_all_conditions() -> None:
    """All conditions in the composite are applied in sequence."""
    mesh = _build_log_spot_mesh()
    put_payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
    cond1 = FdmAmericanStepCondition(mesh, put_payoff)
    composite = FdmStepConditionComposite([], [cond1])
    n = mesh.layout().size()
    a = np.zeros(n, dtype=np.float64)
    composite.apply_to(a, 0.5)
    # American condition floors a[i] to max(K-S, 0).
    locs = mesh.locations(0)
    for i, x in enumerate(locs):
        s = float(np.exp(x))
        expected = max(100.0 - s, 0.0)
        tight(float(a[i]), expected)
