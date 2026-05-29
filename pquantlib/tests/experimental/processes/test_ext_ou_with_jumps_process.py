"""Tests for ExtOUWithJumpsProcess + KlugeExtOUProcess.

# C++ parity: ql/experimental/processes/extouwithjumpsprocess.hpp +
# ql/experimental/processes/klugeextouprocess.hpp.
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.processes.ext_ou_with_jumps_process import (
    ExtOUWithJumpsProcess,
)
from pquantlib.experimental.processes.extended_ornstein_uhlenbeck_process import (
    ExtendedOrnsteinUhlenbeckProcess,
)
from pquantlib.experimental.processes.kluge_ext_ou_process import KlugeExtOUProcess
from pquantlib.testing import tolerance


def _make_ext_ou() -> ExtendedOrnsteinUhlenbeckProcess:
    return ExtendedOrnsteinUhlenbeckProcess(1.0, 0.3, 0.0, lambda _t: 0.0)


def test_ext_ou_with_jumps_accessors() -> None:
    """Accessor round-trip + embedded ExtOU getter.

    EXACT.
    """
    ou = _make_ext_ou()
    p = ExtOUWithJumpsProcess(ou, y0=0.1, beta=4.0, jump_intensity=2.0, eta=5.0)
    tolerance.exact(p.beta(), 4.0)
    tolerance.exact(p.eta(), 5.0)
    tolerance.exact(p.jump_intensity(), 2.0)
    tolerance.exact(p.y0(), 0.1)
    assert p.get_extended_ornstein_uhlenbeck_process() is ou


@pytest.mark.parametrize(
    ("y0", "beta", "lam", "eta"),
    [
        (0.0, -1.0, 1.0, 1.0),  # negative beta
        (0.0, 1.0, -1.0, 1.0),  # negative lambda
        (0.0, 1.0, 1.0, 0.0),  # zero eta
        (0.0, 1.0, 1.0, -1.0),  # negative eta
    ],
)
def test_ext_ou_with_jumps_rejects_invalid_params(
    y0: float, beta: float, lam: float, eta: float
) -> None:
    """Invalid params raise.

    # C++ parity: QL_REQUIRE.
    """
    ou = _make_ext_ou()
    with pytest.raises(LibraryException):
        ExtOUWithJumpsProcess(ou, y0, beta, lam, eta)


def test_kluge_ext_ou_accessors() -> None:
    """KlugeExtOUProcess accessor round-trip.

    EXACT.
    """
    ou_a = _make_ext_ou()
    kluge = ExtOUWithJumpsProcess(ou_a, 0.0, 4.0, 2.0, 5.0)
    ou_b = ExtendedOrnsteinUhlenbeckProcess(0.5, 0.25, 0.0, lambda _t: 0.0)
    p = KlugeExtOUProcess(0.4, kluge, ou_b)
    tolerance.exact(p.rho(), 0.4)
    assert p.get_kluge_process() is kluge
    assert p.get_ext_ou_process() is ou_b


@pytest.mark.parametrize("rho", [-1.5, 1.5, -1.0001, 1.0001])
def test_kluge_ext_ou_rejects_invalid_rho(rho: float) -> None:
    """rho must be in [-1, 1].

    # C++ parity: QL_REQUIRE (rho consistency).
    """
    ou_a = _make_ext_ou()
    kluge = ExtOUWithJumpsProcess(ou_a, 0.0, 4.0, 2.0, 5.0)
    ou_b = _make_ext_ou()
    with pytest.raises(LibraryException):
        KlugeExtOUProcess(rho, kluge, ou_b)
