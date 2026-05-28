"""Tests for ``pquantlib.methods.montecarlo.brownian_bridge.BrownianBridge``.

# C++ parity: ql/methods/montecarlo/brownianbridge.{hpp,cpp} (v1.42.1).

Cross-validates against the ``brownian_bridge_4steps`` section of
``migration-harness/references/cluster/l5c.json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.methods.montecarlo.brownian_bridge import BrownianBridge
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.time_grid import TimeGrid


@pytest.fixture(scope="module")
def reference_data() -> dict[str, Any]:
    return reference_reader.load("cluster/l5c")


def test_bridge_size_steps_constructor() -> None:
    bb = BrownianBridge(4)
    assert bb.size() == 4
    # Unit-spaced grid t = (1, 2, 3, 4).
    assert bb.times() == (1.0, 2.0, 3.0, 4.0)


def test_bridge_from_time_grid_matches_cpp_indices(reference_data: dict[str, Any]) -> None:
    grid = TimeGrid.regular(1.0, 4)
    bb = BrownianBridge.from_time_grid(grid)
    ref = reference_data["brownian_bridge_4steps"]
    assert bb.size() == int(ref["size"])
    # Indices are exact-integer (no tolerance).
    assert list(bb.bridge_index()) == list(ref["bridge_index"])
    assert list(bb.left_index()) == list(ref["left_index"])
    assert list(bb.right_index()) == list(ref["right_index"])


def test_bridge_from_time_grid_matches_cpp_weights(reference_data: dict[str, Any]) -> None:
    grid = TimeGrid.regular(1.0, 4)
    bb = BrownianBridge.from_time_grid(grid)
    ref = reference_data["brownian_bridge_4steps"]
    for got, expected in zip(bb.times(), ref["times"], strict=True):
        tight(got, float(expected))
    for got, expected in zip(bb.std_deviation(), ref["std_deviation"], strict=True):
        tight(got, float(expected))
    for got, expected in zip(bb.left_weight(), ref["left_weight"], strict=True):
        tight(got, float(expected))
    for got, expected in zip(bb.right_weight(), ref["right_weight"], strict=True):
        tight(got, float(expected))


def test_transform_zeros_returns_zeros(reference_data: dict[str, Any]) -> None:
    grid = TimeGrid.regular(1.0, 4)
    bb = BrownianBridge.from_time_grid(grid)
    out = bb.transform(np.zeros(bb.size(), dtype=np.float64))
    expected = reference_data["brownian_bridge_transform_zeros"]
    for got, exp in zip(out, expected, strict=True):
        tight(float(got), float(exp))


def test_transform_impulse_matches_cpp(reference_data: dict[str, Any]) -> None:
    grid = TimeGrid.regular(1.0, 4)
    bb = BrownianBridge.from_time_grid(grid)
    variates = np.zeros(bb.size(), dtype=np.float64)
    variates[0] = 1.0
    out = bb.transform(variates)
    expected = reference_data["brownian_bridge_transform_impulse"]
    for got, exp in zip(out, expected, strict=True):
        tight(float(got), float(exp))


def test_from_times_constructor() -> None:
    """Custom times constructor mirrors C++ ``BrownianBridge(vector<Time>)``."""
    bb = BrownianBridge.from_times([0.25, 0.5, 0.75, 1.0])
    bb2 = BrownianBridge.from_time_grid(TimeGrid.regular(1.0, 4))
    assert bb.std_deviation() == bb2.std_deviation()
    assert bb.bridge_index() == bb2.bridge_index()


def test_transform_size_mismatch_rejected() -> None:
    bb = BrownianBridge(4)
    with pytest.raises(Exception, match="incompatible sequence size"):
        bb.transform(np.zeros(3, dtype=np.float64))
