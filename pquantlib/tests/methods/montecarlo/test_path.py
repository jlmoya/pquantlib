"""Tests for ``pquantlib.methods.montecarlo.path.Path``.

# C++ parity: ql/methods/montecarlo/path.hpp (v1.42.1).
"""

from __future__ import annotations

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.methods.montecarlo.path import Path
from pquantlib.time.time_grid import TimeGrid


def _make_grid() -> TimeGrid:
    """4-step regular grid on [0, 1]."""
    return TimeGrid.regular(1.0, 4)


def test_default_values_zero() -> None:
    grid = _make_grid()
    path = Path(grid)
    assert path.length() == len(grid)
    for i in range(path.length()):
        assert path[i] == 0.0


def test_explicit_values() -> None:
    grid = _make_grid()
    vals = np.array([100.0, 105.0, 110.0, 108.0, 112.0], dtype=np.float64)
    path = Path(grid, vals)
    assert path.front() == 100.0
    assert path.back() == 112.0
    assert path.value(2) == 110.0
    assert path.at(3) == 108.0


def test_size_mismatch_raises() -> None:
    grid = _make_grid()
    bad_vals = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(LibraryException, match="different number of times"):
        Path(grid, bad_vals)


def test_time_accessor() -> None:
    grid = _make_grid()
    path = Path(grid)
    # Regular(1.0, 4) → grid points at 0, 0.25, 0.5, 0.75, 1.0.
    assert path.time(0) == 0.0
    assert path.time(2) == 0.5
    assert path.time(4) == 1.0


def test_iteration() -> None:
    grid = _make_grid()
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    path = Path(grid, vals)
    assert list(path) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_values_property_mutable() -> None:
    """Mirrors C++ ``Path::front()&`` mutable accessor — caller can
    write into the underlying ndarray.
    """
    grid = _make_grid()
    path = Path(grid)
    path.values[0] = 99.0
    assert path.front() == 99.0
