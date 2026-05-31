"""Tests for ExtendedOrnsteinUhlenbeckProcess.

# C++ parity: ql/experimental/processes/extendedornsteinuhlenbeckprocess.hpp.
"""

from __future__ import annotations

import math

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.processes.extended_ornstein_uhlenbeck_process import (
    ExtendedOrnsteinUhlenbeckProcess,
)
from pquantlib.testing import tolerance


def test_extended_ou_accessors() -> None:
    """Constructor + accessor round-trips.

    EXACT: scalar inspectors are bit-identical to ctor inputs.
    """
    p = ExtendedOrnsteinUhlenbeckProcess(2.0, 0.3, 0.1, lambda t: 0.05 * t)
    tolerance.exact(p.x0(), 0.1)
    tolerance.exact(p.speed(), 2.0)
    tolerance.exact(p.volatility(), 0.3)


def test_extended_ou_drift_diffusion_constant_b() -> None:
    """Drift = speed * (b(t) - x); diffusion = sigma.

    TIGHT: closed-form arithmetic.
    """
    b_const = 0.5
    p = ExtendedOrnsteinUhlenbeckProcess(1.0, 0.2, 0.3, lambda _t: b_const)
    tolerance.tight(p.drift_1d(0.5, 0.2), 1.0 * (b_const - 0.2))
    tolerance.tight(p.drift_1d(2.0, 0.6), 1.0 * (b_const - 0.6))
    tolerance.tight(p.diffusion_1d(0.5, 0.2), 0.2)


def test_extended_ou_b_time_dependent() -> None:
    """Time-dependent b: b(t) = sin(t).

    TIGHT: trig closed-form.
    """
    p = ExtendedOrnsteinUhlenbeckProcess(0.8, 0.15, 0.0, math.sin)
    tolerance.tight(p.b(0.0), math.sin(0.0))
    tolerance.tight(p.b(0.5), math.sin(0.5))
    tolerance.tight(p.drift_1d(0.5, 0.1), 0.8 * (math.sin(0.5) - 0.1))


def test_extended_ou_rejects_negative_sigma() -> None:
    """Negative volatility raises.

    # C++ parity: QL_REQUIRE in constructor.
    """
    with pytest.raises(LibraryException):
        ExtendedOrnsteinUhlenbeckProcess(1.0, -0.1, 0.0, lambda _t: 0.0)
