"""Tests for ``pquantlib.methods.montecarlo.path_pricer.PathPricer``.

# C++ parity: ql/methods/montecarlo/pathpricer.hpp (v1.42.1).
"""

from __future__ import annotations

import numpy as np

from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.time.time_grid import TimeGrid


class _TerminalCallPricer(PathPricer[Path]):
    """Trivial concrete pricer: max(S_T - K, 0) — used to test the base."""

    def __init__(self, strike: float, discount: float = 1.0) -> None:
        self._strike = strike
        self._discount = discount

    def __call__(self, path: Path) -> float:
        return self._discount * max(path.back() - self._strike, 0.0)


def test_path_pricer_subclass_callable() -> None:
    pricer = _TerminalCallPricer(strike=100.0, discount=0.95)
    grid = TimeGrid.regular(1.0, 4)
    path = Path(grid, np.array([100.0, 105.0, 110.0, 108.0, 112.0]))
    assert pricer(path) == 0.95 * (112.0 - 100.0)


def test_path_pricer_otm_returns_zero() -> None:
    pricer = _TerminalCallPricer(strike=100.0)
    grid = TimeGrid.regular(1.0, 4)
    path = Path(grid, np.array([100.0, 95.0, 90.0, 92.0, 88.0]))
    assert pricer(path) == 0.0
