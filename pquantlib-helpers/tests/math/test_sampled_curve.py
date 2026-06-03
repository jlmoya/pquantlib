"""Unit tests for the legacy FD math helpers (SampledCurve + Grid builders)."""

from __future__ import annotations

import math

import numpy as np

from pquantlib_helpers.math.grid import (
    bounded_grid,
    bounded_log_grid,
    centered_grid,
)
from pquantlib_helpers.math.sampled_curve import SampledCurve


class TestBoundedLogGrid:
    """``bounded_log_grid`` matches the Java geometric recurrence."""

    def test_endpoints_and_geometric_spacing(self) -> None:
        g = bounded_log_grid(10.0, 100.0, 4)
        assert len(g) == 5
        assert g[0] == 10.0
        # Built by repeated multiplication by exp(log-spacing).
        edx = math.exp((math.log(100.0) - math.log(10.0)) / 4)
        expected = [10.0]
        for _ in range(4):
            expected.append(expected[-1] * edx)
        assert np.allclose(g, expected, rtol=0, atol=0)

    def test_bounded_and_centered(self) -> None:
        assert np.allclose(bounded_grid(0.0, 4.0, 4), [0.0, 1.0, 2.0, 3.0, 4.0])
        assert np.allclose(centered_grid(5.0, 1.0, 4), [3.0, 4.0, 5.0, 6.0, 7.0])


class TestSampledCurve:
    """SampledCurve centre value + derivatives + clone aliasing."""

    def test_value_and_derivatives_at_center_odd(self) -> None:
        c = SampledCurve(np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        c.set_values(np.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=np.float64))  # x^2
        assert c.value_at_center() == 4.0
        # central first derivative at x=2: (9-1)/(3-1) = 4
        assert c.first_derivative_at_center() == 4.0
        # central second derivative ~ 2
        assert math.isclose(c.second_derivative_at_center(), 2.0)

    def test_clone_is_shallow_and_scale_grid_is_in_place(self) -> None:
        # Java parity: clone() shares the grid array; scale_grid mutates in place,
        # so a scale through one curve is visible through its shallow clone — the
        # load-bearing aliasing the FD multi-period engine depends on.
        base = SampledCurve(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        shallow = base.clone()
        base.scale_grid(2.0)
        assert np.allclose(shallow.grid(), [2.0, 4.0, 6.0])

    def test_copy_constructor_is_deep(self) -> None:
        base = SampledCurve(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        deep = SampledCurve(base)
        base.scale_grid(2.0)
        assert np.allclose(deep.grid(), [1.0, 2.0, 3.0])

    def test_set_log_grid_unshares(self) -> None:
        base = SampledCurve(3)
        shallow = base.clone()
        base.set_log_grid(10.0, 40.0)
        # set_log_grid replaces (un-shares) base's grid; the clone keeps the old.
        assert np.allclose(shallow.grid(), [0.0, 0.0, 0.0])
        assert base.grid()[0] == 10.0
