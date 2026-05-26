"""Cross-validate 1-D integrators against the L1-C cluster C++ probe.

Probe source: migration-harness/cpp/probes/cluster_c/probe.cpp
Reference:    migration-harness/references/cluster/c.json

Integrands:
- ``f(x) = x^2`` over [0, 1] = 1/3 (used by all 5 integrators).
- ``f(x) = sin(x)`` over [0, π] = 2 (Simpson / Trapezoid / Kronrod).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.integrals.kronrod import GaussKronrodAdaptive
from pquantlib.math.integrals.lobatto import GaussLobattoIntegral
from pquantlib.math.integrals.segment import SegmentIntegral
from pquantlib.math.integrals.simpson import SimpsonIntegral
from pquantlib.math.integrals.trapezoid import TrapezoidIntegral
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/c")


def _x_squared(x: float) -> float:
    return x * x


# --- integrand: x^2 over [0, 1] = 1/3 ----------------------------------


def test_simpson_x_squared(cpp: dict[str, Any]) -> None:
    simpson = SimpsonIntegral(1e-10, 100)
    value = simpson(_x_squared, 0.0, 1.0)
    tolerance.tight(value, float(cpp["integrals"]["simpson_x_squared"]))


def test_trapezoid_x_squared(cpp: dict[str, Any]) -> None:
    trap = TrapezoidIntegral(1e-10, 100)
    value = trap(_x_squared, 0.0, 1.0)
    tolerance.tight(value, float(cpp["integrals"]["trapezoid_x_squared"]))


def test_segment_x_squared(cpp: dict[str, Any]) -> None:
    seg = SegmentIntegral(100)
    value = seg(_x_squared, 0.0, 1.0)
    tolerance.tight(value, float(cpp["integrals"]["segment_x_squared"]))


def test_kronrod_x_squared(cpp: dict[str, Any]) -> None:
    kr = GaussKronrodAdaptive(1e-10, 100)
    value = kr(_x_squared, 0.0, 1.0)
    tolerance.tight(value, float(cpp["integrals"]["kronrod_x_squared"]))


def test_lobatto_x_squared(cpp: dict[str, Any]) -> None:
    lob = GaussLobattoIntegral(100, 1e-10)
    value = lob(_x_squared, 0.0, 1.0)
    tolerance.tight(value, float(cpp["integrals"]["lobatto_x_squared"]))


# --- integrand: sin(x) over [0, π] = 2 --------------------------------


def test_simpson_sin(cpp: dict[str, Any]) -> None:
    simpson = SimpsonIntegral(1e-10, 100)
    value = simpson(math.sin, 0.0, math.pi)
    tolerance.tight(value, float(cpp["integrals"]["simpson_sin"]))


def test_trapezoid_sin(cpp: dict[str, Any]) -> None:
    trap = TrapezoidIntegral(1e-10, 100)
    value = trap(math.sin, 0.0, math.pi)
    tolerance.tight(value, float(cpp["integrals"]["trapezoid_sin"]))


def test_kronrod_sin(cpp: dict[str, Any]) -> None:
    kr = GaussKronrodAdaptive(1e-10, 100)
    value = kr(math.sin, 0.0, math.pi)
    tolerance.tight(value, float(cpp["integrals"]["kronrod_sin"]))


# --- closed-form sanity / behavior tests -------------------------------


def test_simpson_zero_length_returns_zero() -> None:
    """Integration over a zero-length interval is zero."""
    simpson = SimpsonIntegral(1e-10, 100)
    assert simpson(_x_squared, 1.0, 1.0) == 0.0


def test_simpson_reversed_limits_flips_sign() -> None:
    """``integrate(f, b, a) == -integrate(f, a, b)``."""
    simpson = SimpsonIntegral(1e-10, 100)
    forward = simpson(_x_squared, 0.0, 1.0)
    reverse = simpson(_x_squared, 1.0, 0.0)
    tolerance.tight(reverse, -forward)


def test_segment_zero_intervals_raises() -> None:
    """``SegmentIntegral(0)`` is rejected."""
    with pytest.raises(LibraryException):
        SegmentIntegral(0)


def test_kronrod_max_eval_under_15_raises() -> None:
    """Adaptive Kronrod requires ``max_evaluations >= 15``."""
    with pytest.raises(LibraryException):
        GaussKronrodAdaptive(1e-10, 10)


def test_integrator_rejects_too_small_accuracy() -> None:
    """The Integrator base requires ``absolute_accuracy > QL_EPSILON``."""
    with pytest.raises(LibraryException):
        SimpsonIntegral(1e-20, 100)


def test_simpson_number_of_evaluations_positive() -> None:
    """After a successful call ``number_of_evaluations() > 0``."""
    simpson = SimpsonIntegral(1e-10, 100)
    simpson(_x_squared, 0.0, 1.0)
    assert simpson.number_of_evaluations() > 0


def test_lobatto_constructor_max_first_arg() -> None:
    """``GaussLobattoIntegral(max_evaluations, absolute_accuracy)`` parameter order."""
    lob = GaussLobattoIntegral(200, 1e-10)
    assert lob.max_evaluations() == 200
    assert lob.absolute_accuracy() == 1e-10
