"""Constraint hierarchy behavioral tests.

No C++ probe — the C++ ``Constraint`` family has no probeable
numerical surface (boolean predicates + sentinel-valued bounds).
Tested via direct calls.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.optimization.constraint import (
    BoundaryConstraint,
    CompositeConstraint,
    Constraint,
    NoConstraint,
    NonhomogeneousBoundaryConstraint,
    PositiveConstraint,
)
from pquantlib.testing import reference_reader, tolerance


def _arr(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=np.float64)


def test_no_constraint_accepts_anything() -> None:
    c = NoConstraint()
    assert c.test(_arr(-1.0, 0.0, 1e300))
    assert c.test(_arr(0.0))
    assert c.test(_arr())


def test_no_constraint_default_bounds_are_pm_max() -> None:
    c = NoConstraint()
    params = _arr(0.0, 1.0, 2.0)
    upper = c.upper_bound(params)
    lower = c.lower_bound(params)
    assert np.all(upper == sys.float_info.max)
    assert np.all(lower == -sys.float_info.max)


def test_positive_constraint_requires_strict_positivity() -> None:
    c = PositiveConstraint()
    assert c.test(_arr(1.0, 2.0, 0.5))
    # Zero is rejected (strict positivity).
    assert not c.test(_arr(1.0, 0.0))
    assert not c.test(_arr(-1.0, 1.0))


def test_positive_constraint_lower_bound_is_zero() -> None:
    c = PositiveConstraint()
    params = _arr(1.0, 2.0, 3.0)
    lower = c.lower_bound(params)
    assert np.all(lower == 0.0)
    # Upper bound still defaults to +max.
    upper = c.upper_bound(params)
    assert np.all(upper == sys.float_info.max)


def test_boundary_constraint_inclusive_bounds() -> None:
    c = BoundaryConstraint(low=-1.0, high=2.0)
    assert c.test(_arr(-1.0, 0.0, 2.0))  # boundaries inclusive
    assert not c.test(_arr(-1.5))  # below low
    assert not c.test(_arr(2.1))  # above high


def test_boundary_constraint_bounds_propagate() -> None:
    c = BoundaryConstraint(low=-1.5, high=3.5)
    params = _arr(0.0, 0.0, 0.0)
    assert np.all(c.upper_bound(params) == 3.5)
    assert np.all(c.lower_bound(params) == -1.5)


def test_constraint_is_abstract() -> None:
    with pytest.raises(TypeError, match="Can't instantiate"):
        Constraint()  # type: ignore[abstract]


def test_boundary_constraint_empty_array() -> None:
    c = BoundaryConstraint(low=0.0, high=1.0)
    # Vacuously true.
    assert c.test(_arr())
    assert c.upper_bound(_arr()).shape == (0,)


def test_nonhomogeneous_boundary_constraint_is_per_coordinate() -> None:
    c = NonhomogeneousBoundaryConstraint(_arr(0.0, -1.0, 2.0), _arr(1.0, 1.0, 2.0))
    assert c.test(_arr(0.0, -1.0, 2.0))  # boundaries inclusive
    assert c.test(_arr(0.5, 0.5, 2.0))
    assert not c.test(_arr(-0.1, 0.0, 2.0))  # below its own low
    assert not c.test(_arr(0.5, 1.1, 2.0))  # above its own high


def test_nonhomogeneous_boundary_constraint_bounds_are_the_stored_arrays() -> None:
    low = _arr(0.0, -1.0)
    high = _arr(1.0, 5.0)
    c = NonhomogeneousBoundaryConstraint(low, high)
    # The parameter vector is ignored — C++ returns the stored arrays.
    assert np.array_equal(c.lower_bound(_arr(9.0, 9.0)), low)
    assert np.array_equal(c.upper_bound(_arr(9.0, 9.0)), high)


def test_nonhomogeneous_boundary_constraint_rejects_mismatched_sizes() -> None:
    with pytest.raises(LibraryException, match="boundaries sizes are inconsistent"):
        NonhomogeneousBoundaryConstraint(_arr(0.0, 0.0), _arr(1.0))


def test_nonhomogeneous_boundary_constraint_rejects_wrong_parameter_count() -> None:
    c = NonhomogeneousBoundaryConstraint(_arr(0.0, 0.0), _arr(1.0, 1.0))
    with pytest.raises(LibraryException, match="parameters and boundaries sizes"):
        c.test(_arr(0.5))


# --- CompositeConstraint + Constraint.update (v1.43 additions) --------------


def _cpp_optimization() -> dict[str, Any]:
    return reference_reader.load("v143/math/optimization")


def test_composite_constraint_test_is_the_conjunction() -> None:
    """# C++ parity: constraint.hpp:145-147 — short-circuiting ``&&``."""
    cpp = _cpp_optimization()
    comp = CompositeConstraint(PositiveConstraint(), BoundaryConstraint(-1.0, 2.0))
    assert comp.test(np.array([0.5, 1.5])) is bool(cpp["composite_test_both_ok"])
    assert comp.test(np.array([-0.5, 1.5])) is bool(
        cpp["composite_test_violates_positive"]
    )
    assert comp.test(np.array([0.5, 3.0])) is bool(
        cpp["composite_test_violates_boundary"]
    )


def test_composite_constraint_bounds_are_the_tightest_of_the_two() -> None:
    """upper = elementwise min, lower = elementwise max.

    # C++ parity: constraint.hpp:148-165.
    """
    cpp = _cpp_optimization()
    comp = CompositeConstraint(PositiveConstraint(), BoundaryConstraint(-1.0, 2.0))
    probe = np.array([0.5, 1.5])
    for got, expected in zip(comp.upper_bound(probe), cpp["composite_upper"], strict=True):
        tolerance.exact(float(got), float(expected))
    for got, expected in zip(comp.lower_bound(probe), cpp["composite_lower"], strict=True):
        tolerance.exact(float(got), float(expected))


def test_composite_constraints_nest() -> None:
    cpp = _cpp_optimization()
    comp = CompositeConstraint(PositiveConstraint(), BoundaryConstraint(-1.0, 2.0))
    nested = CompositeConstraint(comp, BoundaryConstraint(0.25, 5.0))
    assert nested.test(np.array([0.5, 1.5])) is bool(cpp["composite_nested_test_ok"])
    assert nested.test(np.array([0.1, 1.5])) is bool(cpp["composite_nested_test_low"])
    probe = np.array([0.5, 1.5])
    for got, expected in zip(nested.upper_bound(probe), cpp["composite_nested_upper"], strict=True):
        tolerance.exact(float(got), float(expected))
    for got, expected in zip(nested.lower_bound(probe), cpp["composite_nested_lower"], strict=True):
        tolerance.exact(float(got), float(expected))


def test_composite_of_no_constraints_keeps_the_open_bounds() -> None:
    """min/max over two +/-DBL_MAX defaults leaves them unchanged."""
    cpp = _cpp_optimization()
    comp = CompositeConstraint(NoConstraint(), NoConstraint())
    probe = np.array([0.5, 1.5])
    for got, expected in zip(comp.upper_bound(probe), cpp["composite_open_upper"], strict=True):
        tolerance.exact(float(got), float(expected))
    for got, expected in zip(comp.lower_bound(probe), cpp["composite_open_lower"], strict=True):
        tolerance.exact(float(got), float(expected))
