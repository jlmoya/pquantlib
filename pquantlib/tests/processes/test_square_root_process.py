"""Tests for SquareRootProcess (CIR diffusion).

C++ parity: ql/processes/squarerootprocess.{hpp,cpp} @ v1.42.1.

No probe needed — drift/diffusion are a trivial closed form.
"""

from __future__ import annotations

import math

from pquantlib.processes.square_root_process import SquareRootProcess
from pquantlib.testing.tolerance import tight


def _process() -> SquareRootProcess:
    # b (mean), a (speed), sigma, x0
    return SquareRootProcess(0.09, 1.0, 0.2, 0.09)


def test_accessors() -> None:
    p = _process()
    tight(p.b(), 0.09)
    tight(p.a(), 1.0)
    tight(p.sigma(), 0.2)
    tight(p.x0(), 0.09)


def test_drift() -> None:
    p = _process()
    # a * (b - x)
    tight(p.drift_1d(0.0, 0.05), 1.0 * (0.09 - 0.05))


def test_diffusion() -> None:
    p = _process()
    # sigma * sqrt(x)
    tight(p.diffusion_1d(0.0, 0.04), 0.2 * math.sqrt(0.04))


def test_mean_reverts_toward_b() -> None:
    p = _process()
    # below mean -> positive drift; above mean -> negative drift
    assert p.drift_1d(0.0, 0.05) > 0.0
    assert p.drift_1d(0.0, 0.15) < 0.0
