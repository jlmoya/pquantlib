"""Tests for ``pquantlib.methods.montecarlo.multi_path.MultiPath``.

# C++ parity: ql/methods/montecarlo/multipath.hpp (v1.42.1).
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.path import Path
from pquantlib.time.time_grid import TimeGrid


def test_from_assets_and_grid_creates_independent_paths() -> None:
    grid = TimeGrid.regular(1.0, 4)
    mp = MultiPath.from_assets_and_grid(3, grid)
    assert mp.asset_number() == 3
    assert mp.path_size() == 5  # 0 + 4 sub-points
    # Each path should be independent (different identity).
    assert mp[0] is not mp[1]
    assert mp[1] is not mp[2]


def test_zero_assets_rejected() -> None:
    grid = TimeGrid.regular(1.0, 4)
    with pytest.raises(LibraryException, match="number of asset"):
        MultiPath.from_assets_and_grid(0, grid)


def test_from_explicit_paths() -> None:
    grid = TimeGrid.regular(1.0, 4)
    paths = [Path(grid), Path(grid)]
    mp = MultiPath(paths)
    assert len(mp) == 2
    assert mp[0] is paths[0]
    assert mp[1] is paths[1]


def test_iteration() -> None:
    grid = TimeGrid.regular(1.0, 4)
    mp = MultiPath.from_assets_and_grid(3, grid)
    items = list(mp)
    assert len(items) == 3
    for p in items:
        assert isinstance(p, Path)
