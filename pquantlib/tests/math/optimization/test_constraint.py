"""Constraint hierarchy behavioral tests.

No C++ probe — the C++ ``Constraint`` family has no probeable
numerical surface (boolean predicates + sentinel-valued bounds).
Tested via direct calls.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from pquantlib.math.optimization.constraint import (
    BoundaryConstraint,
    Constraint,
    NoConstraint,
    PositiveConstraint,
)


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
